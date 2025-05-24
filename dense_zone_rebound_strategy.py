# dense_zone_rebound_strategy.py

import logging
import pandas as pd
from datetime import datetime, timedelta
import asyncio # For async operations

# --- TQSDK Imports ---
from tqsdk import TqApi, TqAuth
# TqAccount and TargetPosTask might be used later for more advanced features.
# from tqsdk import TqAccount, TargetPosTask 

# --- Strategy Parameters ---
# == Dense Trading Zone Parameters ==
ZONE_PRICE_UNIT_TYPE = "tick_multiple" 
ZONE_PRICE_DELTA_UNITS = 5       
ZONE_MIN_REBOUNDS = 2 
ZONE_KLINE_PERIOD_SECONDS = 300  # 5-min klines
ZONE_KLINE_COUNT_REFERENCE = 50 
ZONE_IDENTIFICATION_WINDOW = 20  

# == Smooth Movement Detection Parameters == 
SMOOTH_KLINE_COUNT = 5           
SMOOTH_MIN_DIRECTIONAL_KLINE_RATIO = 0.6 
SMOOTH_MAX_RETRACTMENT_RATIO = 0.5    
SMOOTH_MIN_EFFICIENCY_RATIO = 0.4     

# == Trading Parameters ==
INSTRUMENT_SYMBOL = "SHFE.au2412" # Example, ensure this is a valid and active contract
TRADE_VOLUME = 1                   
ORDER_TYPE = "MARKET" # Strategy currently generates signals, execution uses this type.
SLIPPAGE_TICKS = 2 # For LIMIT orders, if/when implemented in place_trade_action based on signal

# == Other Parameters ==
LOG_LEVEL = "INFO"                 
KLINE_BUFFER = 10 # Extra klines to fetch for time alignment and ensuring enough data

logger = logging.getLogger(__name__)

# Order purposes (already defined in previous versions, kept for clarity)
ORDER_PURPOSE_OPEN = "OPEN_POSITION"
ORDER_PURPOSE_CLOSE = "CLOSE_POSITION"


# --- Helper Functions for Price Handling ---
def get_tick_size_from_quote(quote_obj) -> float | None:
    """
    从天勤SDK的行情对象 (quote object) 中提取最小变动价位 (tick_size)。

    Args:
        quote_obj (TqApi.Quote): 天勤SDK的实时行情数据对象。

    Returns:
        float | None: 如果成功提取，返回合约的最小变动价位 (float)。
                      如果行情对象无效或 tick_size 无效，则返回 None。
    """
    if quote_obj and not pd.isna(quote_obj.tick_size) and quote_obj.tick_size > 0:
        return quote_obj.tick_size
    return None

def get_price_precision(tick_size: float) -> int:
    """
    根据最小变动价位 (tick_size) 决定价格的显示和计算精度（小数位数）。

    例如，如果 tick_size 为 0.01，则精度为 2；如果为 0.5，则精度为 1；如果为 1，则精度为 0。

    Args:
        tick_size (float): 合约的最小变动价位。

    Returns:
        int: 价格应有的小数位数。
    """
    if tick_size == 0: return 2 
    s = f"{tick_size:.10f}" 
    if '.' in s: return len(s.split('.')[1].rstrip('0')) 
    return 0 

def round_to_tick(price: float, tick_size_val: float) -> float:
    """
    根据合约的最小变动价位 (tick_size) 将给定价格圆整到有效的报价。

    Args:
        price (float): 需要圆整的价格。
        tick_size_val (float): 合约的最小变动价位。
        
    Returns:
        float: 圆整到最近有效tick的价格。
    """
    if tick_size_val == 0: return price 
    precision = get_price_precision(tick_size_val)
    return round(round(price / tick_size_val) * tick_size_val, precision)

# --- Core Strategy Classes ---

class DenseTradingZoneIdentifier:
    """
    密集交易区识别器。

    该类负责根据历史K线数据识别价格在特定区间内多次反弹形成的密集交易区。
    这些区域可能代表潜在的支撑或阻力位。
    主要公共方法: `identify_zones`。
    """
    def __init__(self, price_unit_type: str, price_delta_units: int, min_rebounds: int, 
                 kline_period_seconds: int, kline_count_reference: int, identification_window_size: int):
        """
        初始化密集交易区识别器。

        Args:
            price_unit_type (str): 价格区间的单位类型，当前仅支持 "tick_multiple" (最小变动价位的倍数)。
            price_delta_units (int): 定义区域高度的价格差单位数量 (例如，5个ticks)。
            min_rebounds (int): 在一个候选区域内，价格至少需要完成的完整反弹次数 (简化定义：从下边界到上边界再回到下边界算一次)。
            kline_period_seconds (int): 用于区域识别的K线周期（秒数，例如300代表5分钟线）。(当前主要用于获取K线，具体周期影响由调用者传入的K线决定)
            kline_count_reference (int): 用于获取K线数据的参考长度（回溯K线数量）。(此参数主要影响调用者获取多少K线数据，本类使用 `identification_window_size` 进行实际窗口分析)
            identification_window_size (int): 识别单个密集区时使用的滑动窗口大小（K线数量）。
        """
        self.price_unit_type = price_unit_type 
        self.price_delta_units = price_delta_units
        self.min_rebounds = min_rebounds
        self.kline_period_seconds = kline_period_seconds 
        self.kline_count_reference = kline_count_reference 
        self.identification_window_size = identification_window_size 
        if self.price_unit_type != "tick_multiple":
            logger.error("DenseTradingZoneIdentifier: Init Error - Only 'tick_multiple' supported for ZONE_PRICE_UNIT_TYPE.")
            raise ValueError("Only 'tick_multiple' is supported for ZONE_PRICE_UNIT_TYPE.")

    def identify_zones(self, klines_df: pd.DataFrame, current_tick_size: float) -> list:
        """
        从提供的K线数据中识别密集交易区。

        采用滑动窗口方法，在每个窗口内尝试寻找满足多次反弹条件的潜在价格区域。
        “反弹”在这里被简化定义为：价格先触及候选区域的一个边界，然后触及相对的边界，最后再次回到起始边界。

        Args:
            klines_df (pd.DataFrame): 包含K线数据的Pandas DataFrame，
                                      必需列: 'datetime', 'open', 'high', 'low', 'close'。
                                      K线应按时间升序排列。
            current_tick_size (float): 当前合约的最小变动价位，用于计算区域的实际价格高度。

        Returns:
            list: 一个包含已识别密集区信息的字典列表。每个字典结构如下：
                {
                    "zone_top": float,          # 区域上边界价格
                    "zone_bottom": float,       # 区域下边界价格
                    "rebound_count": int,       # 在此区域内观察到的反弹次数
                    "start_time": datetime,     # 形成此区域的K线窗口的开始时间
                    "end_time": datetime,       # 形成此区域的K线窗口的结束时间
                    "start_index": int,         # 原始K线数据中窗口的起始索引
                    "end_index": int            # 原始K线数据中窗口的结束索引
                }
                如果没有识别到区域，则返回空列表。
        """
        identified_zones = []
        if klines_df is None or len(klines_df) < self.identification_window_size:
            return identified_zones
        target_zone_height = self.price_delta_units * current_tick_size
        for i in range(len(klines_df) - self.identification_window_size + 1):
            window_df = klines_df.iloc[i : i + self.identification_window_size].copy()
            window_df.reset_index(drop=True, inplace=True) # Ensure iloc works as expected within window
            window_start_time, window_end_time = window_df['datetime'].iloc[0], window_df['datetime'].iloc[-1]
            
            # 尝试以窗口内的每个K线的低点作为潜在密集区的底部锚点
            for j in range(len(window_df)):
                potential_bottom_anchor = window_df['low'].iloc[j]
                candidate_zone_bottom = potential_bottom_anchor
                candidate_zone_top = potential_bottom_anchor + target_zone_height
                
                rebound_count = 0
                # 状态机: 0 = 初始状态, 1 = 已触及低点等待高点, 2 = 已触及高点等待低点 (完成一次反弹)
                state = 0 
                for k_idx in range(len(window_df)): # 在当前窗口内检查反弹
                    kline = window_df.iloc[k_idx]
                    touched_low = kline['low'] <= candidate_zone_bottom
                    touched_high = kline['high'] >= candidate_zone_top
                    
                    if state == 0: # 初始或完成一次反弹后
                        if touched_low: state = 1
                        elif touched_high: state = 2 # 如果先触碰高点，则等待低点来开始计数周期
                    elif state == 1: # 已触及低点，正在等待高点
                        if touched_high: state = 2 
                    elif state == 2: # 已触及高点，正在等待低点 (若触及则完成一次反弹)
                        if touched_low: 
                            rebound_count += 1
                            state = 1 # 重置状态，开始寻找下一次高点
                
                if rebound_count >= self.min_rebounds:
                    precision = get_price_precision(current_tick_size)
                    zone_info = {"zone_top": round(candidate_zone_top, precision), 
                                 "zone_bottom": round(candidate_zone_bottom, precision),
                                 "rebound_count": rebound_count, 
                                 "start_time": window_start_time, "end_time": window_end_time,
                                 "start_index": klines_df.index[i + j], # Anchor kline's original index
                                 "end_index": klines_df.index[i + self.identification_window_size - 1]} # Window end index
                    
                    # 简化版重叠区域处理：如果新区域与最后添加的区域在价格和时间上有显著重叠，则可能认为是近似重复
                    is_duplicate_or_too_similar = False
                    if identified_zones:
                        last_zone = identified_zones[-1]
                        price_overlap = abs(last_zone["zone_bottom"] - zone_info["zone_bottom"]) <= current_tick_size and \
                                       abs(last_zone["zone_top"] - zone_info["zone_top"]) <= current_tick_size
                        time_overlap = zone_info["start_index"] < last_zone["end_index"] # Simplified time overlap check
                        if price_overlap and time_overlap:
                            is_duplicate_or_too_similar = True
                    
                    if not is_duplicate_or_too_similar:
                        identified_zones.append(zone_info)
        return identified_zones

class SmoothMovementDetector:
    """
    平稳行情检测器。

    该类用于判断一段K线序列是否表现出“平稳”的单向运动特性，
    主要依据方向一致性、回撤幅度和整体运动效率。
    主要公共方法: `is_movement_smooth`。
    """
    def __init__(self, smooth_kline_count: int, min_directional_kline_ratio: float, 
                 max_retracement_ratio: float, min_efficiency_ratio: float):
        """
        初始化平稳行情检测器。

        Args:
            smooth_kline_count (int): 用于观察平稳行情的K线数量。
            min_directional_kline_ratio (float): 在观察窗口内，K线收盘方向与主趋势方向一致的最小比例 (0.0 到 1.0)。
                                                 例如，0.6 表示至少60%的K线需要同向。
            max_retracement_ratio (float): 单根反向K线的实体长度相对于整个平稳阶段净涨跌幅的最大允许比例。
                                           例如，0.3 表示反向K线实体不能超过净运动的30%。
            min_efficiency_ratio (float): 平稳阶段的净涨跌幅相对于总波幅（窗口内最高点-最低点）的最小比率 (0.0 到 1.0)，
                                          衡量趋势的“效率”或“明确性”。
        """
        self.smooth_kline_count = smooth_kline_count
        self.min_directional_kline_ratio = min_directional_kline_ratio
        self.max_retracement_ratio = max_retracement_ratio
        self.min_efficiency_ratio = min_efficiency_ratio
        if not (0 <= self.min_directional_kline_ratio <= 1): raise ValueError("min_directional_kline_ratio must be 0-1.")
        if not (0 <= self.max_retracement_ratio): logger.warning("max_retracement_ratio normally <=1 for strong moves, but any positive value is allowed.")
        if not (0 <= self.min_efficiency_ratio <= 1): raise ValueError("min_efficiency_ratio must be 0-1.")

    def is_movement_smooth(self, klines_recent_df: pd.DataFrame, movement_direction: str) -> bool:
        """
        判断最近的K线序列是否构成预期的平稳行情。

        Args:
            klines_recent_df (pd.DataFrame): 最近的K线数据 (行数应等于 `self.smooth_kline_count`)。
                                           必需列: 'open', 'high', 'low', 'close'。
            movement_direction (str): 预期的主要运动方向，"UP" (向上) 或 "DOWN" (向下)。

        Returns:
            bool: 如果行情被认为是平稳的，则返回 True，否则返回 False。
        """
        if klines_recent_df is None or len(klines_recent_df) != self.smooth_kline_count: return False
        if not all(col in klines_recent_df.columns for col in ['open', 'high', 'low', 'close']): return False
        
        # 1. 方向一致性检查
        if movement_direction == "UP":
            directional_klines = (klines_recent_df['close'] > klines_recent_df['open']).sum()
        elif movement_direction == "DOWN":
            directional_klines = (klines_recent_df['close'] < klines_recent_df['open']).sum()
        else: 
            logger.warning(f"SmoothnessCheck: Invalid movement_direction: {movement_direction}")
            return False
        if (directional_klines / self.smooth_kline_count) < self.min_directional_kline_ratio: return False
        
        # 2. 计算净运动幅度和检查回撤
        first_open = klines_recent_df['open'].iloc[0]
        last_close = klines_recent_df['close'].iloc[-1]
        net_move = (last_close - first_open) if movement_direction == "UP" else (first_open - last_close)
        if net_move <= 0: return False # 必须在预期方向上有所进展

        for _, kline in klines_recent_df.iterrows():
            retracement_body = 0
            if movement_direction == "UP" and kline['close'] < kline['open']: # 上涨中的阴线
                retracement_body = kline['open'] - kline['close']
            elif movement_direction == "DOWN" and kline['close'] > kline['open']: # 下跌中的阳线
                retracement_body = kline['close'] - kline['open']
            
            if retracement_body > (net_move * self.max_retracement_ratio): # 单根K线回撤过大
                return False
        
        # 3. 整体价格进展效率
        segment_high = klines_recent_df['high'].max()
        segment_low = klines_recent_df['low'].min()
        total_segment_range = segment_high - segment_low
        if total_segment_range == 0: return net_move > 0 # 如果K线完全平坦，只要有净移动就认为高效（例如跳空）
        
        efficiency_ratio = net_move / total_segment_range
        return efficiency_ratio >= self.min_efficiency_ratio

class TradingStrategyLogic:
    """
    交易策略的核心逻辑单元。

    该类封装了策略的完整决策流程，包括：
    1. 检查现有持仓的止损条件。
    2. 在无持仓时，结合密集区识别和平稳行情检测来寻找新的交易机会。
    3. 生成具体的交易动作指令（如开多、开空、平多、平空）。

    通过其 `process_signal` 方法接收最新的市场数据（K线和价格），并输出决策。
    主要公共方法: `process_signal`。
    """
    def __init__(self, zone_identifier: DenseTradingZoneIdentifier, 
                 smooth_detector: SmoothMovementDetector, 
                 trade_volume: int, order_type: str):
        """
        初始化交易策略逻辑单元。

        Args:
            zone_identifier (DenseTradingZoneIdentifier): `DenseTradingZoneIdentifier`的实例，用于识别密集交易区。
            smooth_detector (SmoothMovementDetector): `SmoothMovementDetector`的实例，用于判断价格运动是否平稳。
            trade_volume (int): 每次交易的固定手数。
            order_type (str): 期望的订单类型（例如 "MARKET", "LIMIT"），主要用于生成信号，
                              实际执行时可能根据具体情况（如止损）调整。
        """
        self.zone_identifier = zone_identifier
        self.smooth_detector = smooth_detector
        self.trade_volume = trade_volume
        self.order_type = order_type 
        
        self.current_position: str | None = None  # 当前持仓方向: "LONG", "SHORT", 或 None
        self.entry_price: float = 0.0        # 当前持仓的开仓价格
        self.stop_loss_price: float = 0.0    # 当前持仓的止损价格
        self.active_trade_zone: dict | None = None # 触发当前交易的密集区信息 (字典)
        self.last_trade_kline_index: int = -1 # 上次平仓时的K线索引，用于防止在同一区域立即反复开仓

    def process_signal(self, klines_df: pd.DataFrame, latest_price: float, current_tick_size: float) -> dict:
        """
        处理最新的市场数据，并根据策略逻辑生成交易动作。

        主要流程：
        1. 如果有持仓，首先检查是否触发止损。
        2. 如果无持仓，则尝试识别密集交易区。
        3. 若识别到密集区，判断价格是否从区域外平稳运动至区域边缘，以产生开仓信号。
        4. 开仓信号包含目标入场价、手数、止损价及触发的区域信息。

        Args:
            klines_df (pd.DataFrame): 最新的K线数据序列 (Pandas DataFrame)，
                                      应包含足够历史数据以供 `zone_identifier` 和 `smooth_detector` 分析。
            latest_price (float): 最新的市场价格 (例如，当前K线的收盘价或最新tick价)。
            current_tick_size (float): 当前交易合约的最小变动价位。

        Returns:
            dict: 一个包含建议操作的字典。结构示例：
                - 无操作: `{"action": "NONE", "reason": "说明"}`
                - 开多仓: `{"action": "OPEN_LONG", "price": float, "volume": int, "stop_loss": float, "zone_hit": dict}`
                - 开空仓: `{"action": "OPEN_SHORT", "price": float, "volume": int, "stop_loss": float, "zone_hit": dict}`
                - 平多仓: `{"action": "CLOSE_LONG", "reason": "stop_loss", "price": float, "volume": int}` (当前版本止盈由外部管理或未实现)
                - 平空仓: `{"action": "CLOSE_SHORT", "reason": "stop_loss", "price": float, "volume": int}`
        """
        current_kline_index = klines_df.index[-1] if not klines_df.empty else -1
        precision = get_price_precision(current_tick_size)

        # 1. 检查止损
        if self.current_position == "LONG" and latest_price <= self.stop_loss_price:
            logger.info(f"StrategyLogic: Stop-loss for LONG at {self.stop_loss_price:.{precision}f} (current: {latest_price:.{precision}f}).")
            action_details = {"action": ORDER_PURPOSE_CLOSE + "_LONG", "reason": "stop_loss", "price": latest_price, "volume": self.trade_volume}
            self.current_position, self.active_trade_zone, self.last_trade_kline_index = None, None, current_kline_index
            return action_details
        elif self.current_position == "SHORT" and latest_price >= self.stop_loss_price:
            logger.info(f"StrategyLogic: Stop-loss for SHORT at {self.stop_loss_price:.{precision}f} (current: {latest_price:.{precision}f}).")
            action_details = {"action": ORDER_PURPOSE_CLOSE + "_SHORT", "reason": "stop_loss", "price": latest_price, "volume": self.trade_volume}
            self.current_position, self.active_trade_zone, self.last_trade_kline_index = None, None, current_kline_index
            return action_details

        # 2. 无持仓，寻找开仓机会
        if self.current_position is None:
            identified_zones = self.zone_identifier.identify_zones(klines_df, current_tick_size)
            if not identified_zones: return {"action": "NONE", "reason": "No dense zones identified"}
            
            target_zone = identified_zones[-1] # 简化：选择最新识别的区域

            # 防止在刚平仓的区域（或导致平仓的K线形成的区域）上立即重新开仓
            if self.last_trade_kline_index != -1 and target_zone.get("start_index", -2) <= self.last_trade_kline_index:
                # 并且如果这个区域与上次交易的区域相似
                if self.active_trade_zone and \
                   abs(target_zone["zone_bottom"] - self.active_trade_zone.get("zone_bottom", float('inf'))) < current_tick_size and \
                   abs(target_zone["zone_top"] - self.active_trade_zone.get("zone_top", float('-inf'))) < current_tick_size: # .get for safety
                    logger.info(f"StrategyLogic: Skipping re-entry on recently traded zone (Zone start index: {target_zone.get('start_index')}, Last trade kline index: {self.last_trade_kline_index}).")
                    return {"action": "NONE", "reason": "Skipping re-entry on recently traded zone"}

            if len(klines_df) < self.smooth_detector.smooth_kline_count: 
                return {"action": "NONE", "reason": "Not enough klines for smoothness check"}
            recent_klines = klines_df.iloc[-self.smooth_detector.smooth_kline_count:]

            # 情形1: 价格从上方平稳回落至密集区上沿，预期反弹做多 (LONG entry)
            # 条件: 最新价接近或进入区域上部 + 此前的下跌是平稳的
            # 价格在区域上边界附近（允许向上超出一点点，或在边界内）
            zone_top_check = target_zone['zone_top'] + current_tick_size * 2 
            if latest_price <= zone_top_check and latest_price >= target_zone['zone_bottom']: 
                logger.debug(f"StrategyLogic: Price {latest_price:.{precision}f} near zone top {target_zone['zone_top']:.{precision}f}. Checking for smooth DOWN move prior.")
                if self.smooth_detector.is_movement_smooth(recent_klines, "DOWN"): 
                    self.current_position, self.entry_price = "LONG", latest_price 
                    self.stop_loss_price = target_zone['zone_bottom'] # 止损设在区域下边界 (可考虑再减去一个buffer)
                    self.active_trade_zone, self.last_trade_kline_index = target_zone, current_kline_index
                    logger.info(f"StrategyLogic: OPEN_LONG signal at {self.entry_price:.{precision}f}, SL: {self.stop_loss_price:.{precision}f}. Zone: B={target_zone['zone_bottom']:.{precision}f} T={target_zone['zone_top']:.{precision}f}")
                    return {"action": ORDER_PURPOSE_OPEN + "_LONG", "price": self.entry_price, "volume": self.trade_volume, "stop_loss": self.stop_loss_price, "zone_hit": target_zone}

            # 情形2: 价格从下方平稳反弹至密集区下沿，预期受阻回落做空 (SHORT entry)
            # 条件: 最新价接近或进入区域下部 + 此前的上涨是平稳的
            zone_bottom_check = target_zone['zone_bottom'] - current_tick_size * 2
            if latest_price >= zone_bottom_check and latest_price <= target_zone['zone_top']: 
                logger.debug(f"StrategyLogic: Price {latest_price:.{precision}f} near zone bottom {target_zone['zone_bottom']:.{precision}f}. Checking for smooth UP move prior.")
                if self.smooth_detector.is_movement_smooth(recent_klines, "UP"): 
                    self.current_position, self.entry_price = "SHORT", latest_price
                    self.stop_loss_price = target_zone['zone_top'] # 止损设在区域上边界 (可考虑再增加一个buffer)
                    self.active_trade_zone, self.last_trade_kline_index = target_zone, current_kline_index
                    logger.info(f"StrategyLogic: OPEN_SHORT signal at {self.entry_price:.{precision}f}, SL: {self.stop_loss_price:.{precision}f}. Zone: B={target_zone['zone_bottom']:.{precision}f} T={target_zone['zone_top']:.{precision}f}")
                    return {"action": ORDER_PURPOSE_OPEN + "_SHORT", "price": self.entry_price, "volume": self.trade_volume, "stop_loss": self.stop_loss_price, "zone_hit": target_zone}
        
        return {"action": "NONE", "reason": "No entry conditions met or position already active"}


async def place_trade_action(api: TqApi, instrument_symbol: str, trade_action: dict, current_orders_map: dict):
    """
    根据 `trade_action` 字典中的指令，异步执行实际的下单操作。

    此函数处理开仓和平仓指令，并尝试使用全局 `ORDER_TYPE`（如 "MARKET" 或 "LIMIT"）执行。
    对于平仓操作（通常由止损触发），建议使用市价单以确保成交。

    Args:
        api (TqApi): 天勤API实例。
        instrument_symbol (str): 交易合约代码。
        trade_action (dict): `TradingStrategyLogic.process_signal` 返回的包含操作指令的字典。
                             必需键: "action", "volume"。可选键: "price" (用于限价单)。
        current_orders_map (dict): 一个用于跟踪当前活动订单的字典，键为合约代码，
                                   值为包含订单详情（如order_id, status, order_obj, purpose）的字典。
                                   此函数会更新这个字典以反映新下的订单。

    Returns:
        TqApi.Order | None: 如果下单成功，返回天勤SDK的订单对象。否则返回 None。
    """
    action_type = trade_action.get("action")
    volume = trade_action.get("volume", TRADE_VOLUME) 
    price = trade_action.get("price") 
    
    offset_flag, direction_flag = "", ""
    if action_type == ORDER_PURPOSE_OPEN + "_LONG": direction_flag, offset_flag = "BUY", "OPEN"
    elif action_type == ORDER_PURPOSE_OPEN + "_SHORT": direction_flag, offset_flag = "SELL", "OPEN"
    elif action_type == ORDER_PURPOSE_CLOSE + "_LONG": direction_flag, offset_flag = "SELL", "CLOSE"
    elif action_type == ORDER_PURPOSE_CLOSE + "_SHORT": direction_flag, offset_flag = "BUY", "CLOSE"
    else: 
        logger.warning(f"place_trade_action: Unknown action type: {action_type}")
        return None

    # 防止对同一合约重复提交相同目的的活动订单 (简化版检查)
    if instrument_symbol in current_orders_map:
        existing_order_data = current_orders_map[instrument_symbol]
        if existing_order_data.get("purpose") == action_type and \
           hasattr(existing_order_data.get("order_obj"), "status") and \
           existing_order_data["order_obj"].status == "ALIVE": # TQSDK 订单状态通常是大写
            logger.info(f"place_trade_action: Existing ALIVE order for {instrument_symbol} with purpose {action_type} (ID: {existing_order_data['order_id']}). Skipping new order.")
            return existing_order_data['order_obj'] 

    precision = get_price_precision(get_tick_size(api, instrument_symbol))
    logger.info(f"place_trade_action: Attempting to {action_type} {volume} lots of {instrument_symbol} via {ORDER_TYPE} order (Price context: {price:.{precision}f if isinstance(price, (int,float)) else 'N/A'}).")
    
    order_to_place = None
    try:
        # 对于平仓操作 (SL/TP触发)，通常使用市价单以保证成交
        effective_order_type = "MARKET" if "CLOSE" in action_type else ORDER_TYPE

        if effective_order_type == "MARKET":
            order_to_place = await api.insert_order(symbol=instrument_symbol, direction=direction_flag, offset=offset_flag, volume=volume, order_type="MARKET")
        elif effective_order_type == "LIMIT": 
            limit_price = price 
            if not isinstance(limit_price, (int, float)): 
                logger.error(f"place_trade_action: Cannot place LIMIT order for {instrument_symbol} without a valid price for action {action_type}. Price provided: {limit_price}")
                return None
            order_to_place = await api.insert_order(symbol=instrument_symbol, direction=direction_flag, offset=offset_flag, volume=volume, order_type="LIMIT", limit_price=limit_price)
        else:
            logger.error(f"place_trade_action: Unsupported ORDER_TYPE: {effective_order_type}"); return None
        
        if order_to_place and hasattr(order_to_place, 'order_id') and order_to_place.order_id:
             current_orders_map[instrument_symbol] = {"order_id": order_to_place.order_id, 
                                                      "status": "ALIVE", # 初始状态假设为 ALIVE
                                                      "purpose": action_type, 
                                                      "order_obj": order_to_place}
             logger.info(f"place_trade_action: Order {order_to_place.order_id} placed for {action_type} on {instrument_symbol}.")
        elif order_to_place: 
             logger.warning(f"place_trade_action: Order placed for {action_type} on {instrument_symbol}, but order_id is missing or invalid. Order details: {order_to_place}")
        else: 
             logger.error(f"place_trade_action: Failed to place order for {action_type} on {instrument_symbol}. insert_order returned None or invalid object.")
        return order_to_place
    except Exception as e:
        logger.error(f"place_trade_action: Error placing order for {instrument_symbol} ({action_type}): {e}", exc_info=True)
        return None

async def run_live_strategy(api: TqApi):
    """
    运行实盘（或模拟实盘）交易策略的主异步函数。

    负责初始化策略组件、订阅数据、进入主循环处理市场行情、
    生成交易信号、执行交易、管理订单和持仓状态，以及处理风险（止盈止损）。
    该函数会持续运行，直到被外部中断 (例如 `KeyboardInterrupt`)。

    Args:
        api (TqApi): 已初始化的天勤API实例。
    """
    logger.info(f"run_live_strategy started for {INSTRUMENT_SYMBOL}.")
    
    instrument_info = await api.get_instrument_info(INSTRUMENT_SYMBOL)
    if not instrument_info:
        logger.error(f"Could not fetch instrument info for {INSTRUMENT_SYMBOL}. Exiting strategy for this instrument.")
        return
    current_tick_size = instrument_info.tick_price 
    logger.info(f"Instrument: {INSTRUMENT_SYMBOL}, Tick Size: {current_tick_size}")

    # Initialize strategy components
    zone_identifier = DenseTradingZoneIdentifier(
        price_unit_type=ZONE_PRICE_UNIT_TYPE, price_delta_units=ZONE_PRICE_DELTA_UNITS,
        min_rebounds=ZONE_MIN_REBOUNDS, kline_period_seconds=ZONE_KLINE_PERIOD_SECONDS,
        kline_count_reference=ZONE_KLINE_COUNT_REFERENCE, identification_window_size=ZONE_IDENTIFICATION_WINDOW
    )
    smooth_detector = SmoothMovementDetector(
        smooth_kline_count=SMOOTH_KLINE_COUNT, min_directional_kline_ratio=SMOOTH_MIN_DIRECTIONAL_KLINE_RATIO,
        max_retracement_ratio=SMOOTH_MAX_RETRACTMENT_RATIO, min_efficiency_ratio=SMOOTH_MIN_EFFICIENCY_RATIO
    )
    strategy_logic = TradingStrategyLogic( 
        zone_identifier=zone_identifier, smooth_detector=smooth_detector,
        trade_volume=TRADE_VOLUME, order_type=ORDER_TYPE 
    )
    
    active_tq_orders_map = {} 

    await api.subscribe_quote(INSTRUMENT_SYMBOL)
    logger.info(f"Subscribed to quotes for {INSTRUMENT_SYMBOL}.")
    
    required_klines_length = ZONE_KLINE_COUNT_REFERENCE + SMOOTH_KLINE_COUNT + KLINE_BUFFER 
    
    # Initial kline fetch
    klines_df = await api.get_kline_serial(INSTRUMENT_SYMBOL, ZONE_KLINE_PERIOD_SECONDS, data_length=required_klines_length)
    if not isinstance(klines_df, pd.DataFrame) or klines_df.empty:
        logger.warning(f"Initial klines fetch for {INSTRUMENT_SYMBOL} failed or returned empty. Strategy may need more data to start.")
    else:
        logger.info(f"Initial klines fetched for {INSTRUMENT_SYMBOL}: {len(klines_df)} rows.")

    log_counter = 0 
    try:
        while True:
            await api.wait_update() 
            log_counter +=1

            klines_df_updated = await api.get_kline_serial(INSTRUMENT_SYMBOL, ZONE_KLINE_PERIOD_SECONDS, required_klines_length)
            quote = await api.get_quote(INSTRUMENT_SYMBOL)

            if not isinstance(klines_df_updated, pd.DataFrame) or klines_df_updated.empty or quote is None or pd.isna(quote.last_price):
                logger.warning("Could not get valid klines or quote in main loop. Skipping this cycle.")
                continue
            
            latest_price = quote.last_price
            klines_df_updated.reset_index(inplace=True, drop=True) 

            action = strategy_logic.process_signal(klines_df_updated, latest_price, current_tick_size)

            if action and action.get("action") != "NONE":
                logger.info(f"Strategy generated action: {action}")
                await place_trade_action(api, INSTRUMENT_SYMBOL, action, active_tq_orders_map)
            
            for sym_in_map, order_details in list(active_tq_orders_map.items()):
                tq_order_obj = order_details.get("order_obj")
                if tq_order_obj and tq_order_obj.is_finished(): 
                    logger.info(f"Live Order Monitor: Order {tq_order_obj.order_id} for {sym_in_map} (Purpose: {order_details.get('purpose')}) is finished. Status: {tq_order_obj.status_message}")
                    del active_tq_orders_map[sym_in_map]
            
            if log_counter % 60 == 0: # Periodic logging (e.g., every 60 seconds if wait_update timeout is 1s)
                acc = await api.get_account()
                pos_tq = await api.get_position(INSTRUMENT_SYMBOL) # Actual position from TQSDK
                precision = get_price_precision(current_tick_size)
                logger.info(f"Periodic Update: Strategy State for {INSTRUMENT_SYMBOL}: Position='{strategy_logic.current_position}', Entry={strategy_logic.entry_price:.{precision}f}, SL={strategy_logic.stop_loss_price:.{precision}f}")
                if acc: logger.info(f"  Account: Available={acc.available}, Balance={acc.balance}")
                if pos_tq: logger.info(f"  Actual TQ Position for {INSTRUMENT_SYMBOL}: Long={pos_tq.pos_long}@{pos_tq.open_price_long:.{precision}f}, Short={pos_tq.pos_short}@{pos_tq.open_price_short:.{precision}f}")

    finally: # Cleanup within run_live_strategy
        if not api.is_closed(): # Check if api is still valid
            logger.info(f"Initiating shutdown process within run_live_strategy for {INSTRUMENT_SYMBOL} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
            for sym, order_data in list(active_tq_orders_map.items()): # Cancel pending opening orders
                order = order_data.get("order")
                try:
                    if order and hasattr(order, 'order_id') and hasattr(order, 'status') and \
                       order.status == "ALIVE" and order_data.get("purpose") == ORDER_PURPOSE_OPEN :
                         logger.info(f"  Cancelling ALIVE opening order {order.order_id} for {sym}...")
                         await api.cancel_order(order) # Use await for async cancel if available, TQSDK might handle this internally
                except Exception as cancel_e: 
                    logger.error(f"  Error cancelling order for {sym} (ID: {order.order_id if order else 'N/A'}): {cancel_e}", exc_info=False)
        # api.close() is called in the main __name__ block's finally


if __name__ == "__main__":
    log_level_actual = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    if not logger.handlers: 
        logger.setLevel(log_level_actual)
        logger.addHandler(stream_handler)
        try:
            fh = logging.FileHandler('dense_zone_strategy.log', mode='a') 
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e: logger.error(f"Failed to set up file handler: {e}")
    else: 
        logger.setLevel(log_level_actual)

    logger.info("Dense Zone Rebound Strategy - TQSDK Live Integration Run")
    
    api_instance = None 
    try:
        logger.info("Initializing TQApi...")
        # For real trading, replace TqAuth() with actual account credentials and settings
        api_instance = TqApi(auth=TqAuth()) 
        logger.info(f"TQApi initialized. TQSDK Version: {api_instance.lib_version}, Broker: {api_instance.broker_id if api_instance.broker_id else 'N/A'}")
        account_info = asyncio.run(api_instance.get_account()) # Use asyncio.run for initial async call if outside async func
        if account_info: logger.info(f"Account ID: {account_info.account_id}, Balance: {account_info.balance}, Available: {account_info.available}")
        
        asyncio.run(run_live_strategy(api_instance)) 
    except KeyboardInterrupt:
        logger.info("Strategy execution stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.exception(f"Critical error in main asyncio run: {e}")
    finally:
        if api_instance and not api_instance.is_closed(): 
            logger.info(f"Closing API connection from __main__ at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
            api_instance.close() 
            logger.info("API connection closed.")
        else: 
            logger.info("API was not initialized or already closed.")
        logger.info("Dense Zone Rebound Strategy script finished.")

```
