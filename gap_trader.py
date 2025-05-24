#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from tqsdk import TqApi, TqAuth
import datetime 
import logging 

# --- Global Configuration ---
# Trading parameters
MINIMUM_GAP_PERCENTAGE = 0.5  # Minimum percentage change for a gap to be considered significant
STOP_LOSS_PERCENTAGE = 1.0    # Percentage from entry price to set stop-loss
TAKE_PROFIT_PERCENTAGE = 1.5  # Percentage from entry price to set take-profit
TRADE_VOLUME = 1              # Fixed volume for each trade

# List of instruments to monitor and trade
INSTRUMENTS = ["SHFE.rb2410", "DCE.i2409", "CZCE.MA2409", "INE.sc2407"] 

# Order purposes
ORDER_PURPOSE_OPEN = "OPEN_POSITION"
ORDER_PURPOSE_CLOSE = "CLOSE_POSITION"

# Logger setup (configured in main)
logger = logging.getLogger(__name__)

# --- Helper Functions for Price Handling ---
def get_tick_size(api: TqApi, instrument_symbol: str) -> float:
    """
    获取指定合约的最小变动价位 (tick_size)。
    如果无法从API获取有效的tick_size，则使用预定义的启发式回退值。

    Args:
        api (TqApi): 天勤API实例。
        instrument_symbol (str): 合约代码，例如 "SHFE.rb2410"。

    Returns:
        float: 合约的最小变动价位。
    """
    quote = api.get_quote(instrument_symbol)
    if quote and not pd.isna(quote.tick_size) and quote.tick_size > 0:
        return quote.tick_size
    
    logger.warning(f"无法从API获取 {instrument_symbol} 的有效tick_size，将使用启发式回退值。")
    # Fallback heuristic (can be improved with a predefined map for common instruments)
    if "SHFE.au" in instrument_symbol or "INE.au" in instrument_symbol : return 0.01 
    if "SHFE.ag" in instrument_symbol : return 1.0
    if "SHFE.rb" in instrument_symbol or "SHFE.hc" in instrument_symbol : return 1.0
    if "DCE.i" in instrument_symbol : return 0.5
    if "CZCE.MA" in instrument_symbol: return 1.0
    if "INE.sc" in instrument_symbol: return 0.1
    logger.warning(f"没有为 {instrument_symbol} 设置特定的回退tick_size，默认使用1.0。价格取整可能不准确。")
    return 1.0 

def get_price_precision(tick_size: float) -> int:
    """
    根据最小变动价位 (tick_size) 决定价格的显示精度（小数位数）。

    Args:
        tick_size (float): 合约的最小变动价位。

    Returns:
        int: 价格应显示的小数位数。
    """
    if tick_size == 0: return 2 # Default precision if tick_size is somehow zero
    s = f"{tick_size:.10f}" # Format to string with many decimal places
    if '.' in s:
        return len(s.split('.')[1].rstrip('0')) # Count digits after decimal, removing trailing zeros
    return 0 # No decimal places if it's an integer tick size

def round_to_tick(price: float, tick_size_val: float) -> float:
    """
    根据合约的最小变动价位 (tick_size) 将给定价格圆整到有效的报价。

    Args:
        price (float): 需要圆整的价格。
        tick_size_val (float): 合约的最小变动价位。
        
    Returns:
        float: 圆整到最近有效tick的价格。
    """
    # precision argument removed as it's derived from tick_size_val now
    if tick_size_val == 0: return price 
    precision = get_price_precision(tick_size_val)
    return round(round(price / tick_size_val) * tick_size_val, precision)

# --- Core Trading Logic Functions ---
def check_opening_gap(api: TqApi, instrument_symbol: str) -> dict | None:
    """
    检查指定合约在开盘时是否存在价格缺口。

    通过比较前一交易日的收盘价和当日的开盘价来确定缺口类型和幅度。

    Args:
        api (TqApi): 天勤API实例。
        instrument_symbol (str): 需要检查的合约代码。

    Returns:
        dict | None: 包含缺口详细信息的字典，例如：
            {
                "instrument": str,          # 合约代码
                "previous_close": float,    # 前一交易日收盘价
                "current_open": float,      # 当日开盘价
                "gap_percentage": float,    # 缺口百分比 (正数为向上缺口, 负数为向下缺口)
                "gap_type": str,            # "up-gap", "down-gap", "no_gap", 或错误类型
                "error": str | None         # 如果发生错误，则包含错误信息
            }
            如果无法获取数据或发生严重错误，则返回 None 或包含错误信息的字典。
    """
    try:
        klines = api.get_kline_serial(instrument_symbol, duration_seconds=24 * 60 * 60, data_length=2)
        if klines is None or len(klines) < 2:
            logger.warning(f"Gap Check: Insufficient K-line data for {instrument_symbol} (received {len(klines) if klines is not None else 0} bars).")
            return {"instrument": instrument_symbol, "error": "Insufficient K-line data", "gap_type": "error_data"}
        
        previous_close = klines.iloc[-2]['close']
        current_open = klines.iloc[-1]['open']

        if pd.isna(previous_close) or pd.isna(current_open):
            logger.warning(f"Gap Check: NaN in price data for {instrument_symbol} (PrevClose: {previous_close}, CurrOpen: {current_open}).")
            return {"instrument": instrument_symbol, "error": "NaN in price data", "previous_close": previous_close, "current_open": current_open, "gap_type": "error_data_nan"}
        if previous_close == 0: 
             logger.warning(f"Gap Check: Previous close is zero for {instrument_symbol}, cannot calculate percentage gap.")
             return {"instrument": instrument_symbol, "previous_close": previous_close, "current_open": current_open, "error": "Previous close is zero", "gap_type": "error_zero_close"}
        
        gap_percentage = ((current_open - previous_close) / previous_close) * 100
        gap_type = "no_gap"
        if gap_percentage > MINIMUM_GAP_PERCENTAGE: gap_type = "up-gap"
        elif gap_percentage < -MINIMUM_GAP_PERCENTAGE: gap_type = "down-gap"
        
        return {"instrument": instrument_symbol, "previous_close": previous_close, "current_open": current_open, 
                "gap_percentage": round(gap_percentage, 2), "gap_type": gap_type, "error": None}
    except Exception as e:
        logger.error(f"Gap Check Error for {instrument_symbol}: {e}", exc_info=False)
        return {"instrument": instrument_symbol, "error": str(e), "gap_type": "error_exception"}

def determine_trade_signal(api: TqApi, gap_info: dict | None) -> dict | None:
    """
    根据缺口分析结果生成交易信号，包括预期的交易方向、入场价、止损价和止盈价。

    缺口交易策略通常是反向操作：
    - 向上缺口（价格跳空高开）：预期价格可能回落，产生卖出（做空）信号。
    - 向下缺口（价格跳空低开）：预期价格可能反弹，产生买入（做多）信号。

    Args:
        api (TqApi): 天勤API实例，用于获取合约的tick_size以精确计算价格。
        gap_info (dict | None): `check_opening_gap` 函数返回的缺口信息字典。

    Returns:
        dict | None: 包含交易信号详情的字典，例如：
            {
                "instrument": str,          # 合约代码
                "signal_type": str,         # "BUY_OPEN" (做多开仓) 或 "SELL_SHORT" (做空开仓)
                "entry_price_target": float,# 目标入场价 (通常是当日开盘价)
                "stop_loss_price": float,   # 计算得出的止损价
                "take_profit_price": float, # 计算得出的止盈价
                "reason": str               # 产生信号的原因，如缺口类型和幅度
            }
            如果没有有效的交易信号（例如，没有缺口或数据错误），则返回 None。
    """
    if not gap_info or gap_info.get("error") or not isinstance(gap_info.get("current_open"), (int, float)): 
        # 如果gap_info无效，或其中包含错误，或当日开盘价无效，则不生成信号
        return None
    
    instrument = gap_info["instrument"]
    gap_type = gap_info["gap_type"]
    current_open = gap_info["current_open"]
    gap_percentage = gap_info.get("gap_percentage", 0.0) # 若无gap_percentage则默认为0
    
    tick_size = get_tick_size(api, instrument) # 获取合约的最小变动单位
    entry_price = current_open # 交易信号的入场目标价通常设为当日开盘价

    if gap_type == "up-gap": # 向上缺口，预期价格回落，产生卖出（做空）信号
        signal_type = "SELL_SHORT"
        # 止损设在开盘价上方 (开盘价 * (1 + 百分比))
        stop_loss = round_to_tick(entry_price * (1 + STOP_LOSS_PERCENTAGE / 100), tick_size)
        # 止盈设在开盘价下方 (开盘价 * (1 - 百分比))
        take_profit = round_to_tick(entry_price * (1 - TAKE_PROFIT_PERCENTAGE / 100), tick_size)
    elif gap_type == "down-gap": # 向下缺口，预期价格反弹，产生买入（做多）信号
        signal_type = "BUY_OPEN"
        # 止损设在开盘价下方 (开盘价 * (1 - 百分比))
        stop_loss = round_to_tick(entry_price * (1 - STOP_LOSS_PERCENTAGE / 100), tick_size)
        # 止盈设在开盘价上方 (开盘价 * (1 + 百分比))
        take_profit = round_to_tick(entry_price * (1 + TAKE_PROFIT_PERCENTAGE / 100), tick_size)
    else: # "no_gap" 或其他类型的缺口 (如错误类型)
        return None # 不生成交易信号
        
    return {"instrument": instrument, "signal_type": signal_type, "entry_price_target": entry_price, 
            "stop_loss_price": stop_loss, "take_profit_price": take_profit, 
            "reason": f"{gap_type.replace('-', ' ').title()} of {gap_percentage:.2f}%"}

def execute_trade(api: TqApi, trade_signal: dict | None) -> TqApi.Order | None:
    """
    根据交易信号执行开仓操作。

    本函数会根据 `trade_signal` 中的指示，下达一个限价单 (`LIMIT`) 来尝试开仓。
    开仓手数由全局变量 `TRADE_VOLUME` 定义。

    Args:
        api (TqApi): 天勤API实例。
        trade_signal (dict | None): `determine_trade_signal` 函数返回的交易信号字典。
                                   该字典必须包含 "instrument", "signal_type", 和 "entry_price_target"。

    Returns:
        TqApi.Order | None: 如果下单成功，返回天勤SDK的订单对象 (TqApi.Order)。
                            如果 `trade_signal` 无效或下单过程中发生错误，则返回 None。
    """
    # active_orders argument removed as it's managed within run_bot now
    if not trade_signal: return None
    
    instrument = trade_signal["instrument"]
    signal_type = trade_signal["signal_type"]
    entry_price_target = trade_signal["entry_price_target"]
    
    direction, offset = (None, None)
    if signal_type == "BUY_OPEN": direction, offset = "BUY", "OPEN"
    elif signal_type == "SELL_SHORT": direction, offset = "SELL", "OPEN"
    
    if not direction or not isinstance(entry_price_target, (int, float)): 
        logger.warning(f"Execute Trade: Invalid signal or price for {instrument}. Signal: {signal_type}, Price: {entry_price_target}")
        return None
        
    precision = get_price_precision(get_tick_size(api, instrument))
    logger.info(f"Execute Trade: Attempting {direction} {offset} {TRADE_VOLUME} of {instrument} at {entry_price_target:.{precision}f} (LIMIT).")
    try:
        order = api.insert_order(symbol=instrument, direction=direction, offset=offset, 
                                 volume=TRADE_VOLUME, limit_price=entry_price_target, order_type="LIMIT")
        return order 
    except Exception as e: 
        logger.error(f"Execute Trade Error for {instrument}: {e}", exc_info=False)
        return None

def close_position(api: TqApi, instrument_symbol: str, pos_details: dict) -> TqApi.Order | None:
    """
    对指定合约的现有持仓执行平仓操作。

    本函数会根据 `pos_details` 中的持仓方向和手数，下达一个市价单 (`MARKET`) 来平仓。

    Args:
        api (TqApi): 天勤API实例。
        instrument_symbol (str): 需要平仓的合约代码。
        pos_details (dict): 包含当前持仓详细信息的字典，至少需要:
                            `"direction"`: "LONG" 或 "SHORT"
                            `"volume"`: 持仓手数

    Returns:
        TqApi.Order | None: 如果下单成功，返回天勤SDK的订单对象 (TqApi.Order)。
                            如果下单过程中发生错误，则返回 None。
    """
    # active_orders argument removed as it's managed within run_bot now
    direction_to_close = "SELL" if pos_details["direction"] == "LONG" else "BUY"
    volume_to_close = pos_details["volume"]
    logger.info(f"Close Position: Attempting {direction_to_close} CLOSE {volume_to_close} of {instrument_symbol} (MARKET).")
    try:
        # TQSDK的 "CLOSE" offset 会自动处理期货的平今/平昨问题
        closing_order = api.insert_order(symbol=instrument_symbol, direction=direction_to_close, 
                                         offset="CLOSE", volume=volume_to_close, order_type="MARKET")
        return closing_order
    except Exception as e: 
        logger.error(f"Close Position Error for {instrument_symbol}: {e}", exc_info=False)
        return None

# --- Main Application Logic ---
def run_bot(api: TqApi):
    """
    运行交易机器人的主逻辑。

    包括初始化、订阅行情、检查初始持仓、在开盘时进行一次缺口分析和潜在交易、
    进入主监控循环（处理订单更新、持仓更新、风险管理如止盈止损），以及退出时的清理。

    Args:
        api (TqApi): 已初始化的天勤API实例。
    """
    # State tracking dictionaries
    # active_orders: instrument_symbol -> {"order": TqApi.Order, "sl_price": float, "tp_price": float, "purpose": str}
    active_orders = {}     
    # current_positions: instrument_symbol -> {"direction": str, "volume": int, "open_price": float, "sl_price": float, "tp_price": float, "update_time": datetime, "order_id_open": str | None}
    current_positions = {} 
    # filled_opening_order_details: order_id -> {"sl_price": float, "tp_price": float, "volume_filled": int, "instrument_id": str}
    # 此字典用于暂存已成交开仓订单的止盈止损信息，以便在持仓更新时能准确关联。
    filled_opening_order_details = {} 

    try:
        logger.info("Subscribing to quotes for all instruments...")
        for sym in INSTRUMENTS: 
            api.subscribe_quote(sym)
            logger.info(f"  Subscribed quotes for {sym}.")
        
        logger.info("Performing initial position check...")
        for sym in INSTRUMENTS:
            pos = api.get_position(sym)
            if pos and (pos.pos_long > 0 or pos.pos_short > 0):
                direction = "LONG" if pos.pos_long > 0 else "SHORT"
                volume = pos.pos_long or pos.pos_short
                open_price = pos.open_price_long or pos.open_price_short
                current_positions[sym] = {"direction": direction, "volume": volume, "open_price": open_price, 
                                        "sl_price": None, "tp_price": None, "update_time": api.get_time(),
                                        "order_id_open": None} # 对于已存在的持仓，开仓订单ID未知
                precision = get_price_precision(get_tick_size(api, sym))
                logger.info(f"  Initial Position: {direction} {volume} of {sym} at {open_price:.{precision}f} (SL/TP not script-defined).")
        logger.info("Initial position check complete.\n" + "="*50)

        # One-time gap analysis and initial trade placement at startup
        logger.info("Analyzing gaps for initial trades...")
        for sym in INSTRUMENTS:
            logger.info(f"  Processing for Gap Analysis: {sym}")
            gap_info = check_opening_gap(api, sym)
            precision = get_price_precision(get_tick_size(api, sym)) # For logging prices accurately
            if gap_info and not gap_info.get("error"):
                logger.info(f"    Gap Analysis for {sym}: PrevClose: {gap_info.get('previous_close')}, Open: {gap_info.get('current_open')}, %Gap: {gap_info.get('gap_percentage')}, Type: {gap_info.get('gap_type')}")
                trade_signal = determine_trade_signal(api, gap_info)
                if trade_signal:
                    logger.info(f"    Trade Signal for {sym}: {trade_signal['signal_type']} at {trade_signal['entry_price_target']:.{precision}f}, SL: {trade_signal['stop_loss_price']:.{precision}f}, TP: {trade_signal['take_profit_price']:.{precision}f}")
                    if sym in current_positions: 
                        logger.info(f"    Action: Skip trade, existing position for {sym}.")
                    elif sym in active_orders and active_orders[sym]["purpose"] == ORDER_PURPOSE_OPEN: 
                        logger.info(f"    Action: Skip trade, active opening order for {sym} (ID: {active_orders[sym]['order'].order_id if active_orders[sym].get('order') else 'N/A'}).")
                    else:
                        order_obj = execute_trade(api, trade_signal)
                        if order_obj: 
                            active_orders[sym] = {"order": order_obj, 
                                                "sl_price": trade_signal["stop_loss_price"], 
                                                "tp_price": trade_signal["take_profit_price"], 
                                                "purpose": ORDER_PURPOSE_OPEN}
                            logger.info(f"    Action: Opening order (ID: {order_obj.order_id}) sent for {sym}.")
                else: 
                    logger.info(f"    Trade Signal: No signal for {sym} based on gap analysis.")
            else: 
                logger.warning(f"    Gap Analysis Error for {sym}: {gap_info.get('error', 'N/A') if gap_info else 'N/A'}")
            logger.info("-"*40) # Separator for each instrument's initial analysis
        logger.info("Initial gap analysis and trade placement complete.\n" + "="*50)

        logger.info(f"Bot monitoring loop started: {api.get_time().strftime('%Y-%m-%d %H:%M:%S')}")
        while True:
            api.wait_update(deadline=api.get_time() + pd.Timedelta(seconds=1)) # Wait for updates with 1s timeout

            # --- 1. Process Active Orders ---
            for sym, order_data in list(active_orders.items()): # Iterate copy for safe removal
                order_obj = order_data["order"]
                if order_obj.is_finished():
                    logger.info(f"  Order Monitor: Order for {sym} (ID: {order_obj.order_id}, Purpose: {order_data['purpose']}) FINISHED. Status: {order_obj.status_message}, Vol Traded: {order_obj.volume_traded}/{order_obj.volume_orign}")
                    if order_data["purpose"] == ORDER_PURPOSE_OPEN and order_obj.volume_traded > 0:
                        # Store SL/TP info for filled opening order to be picked up by position update logic
                        filled_opening_order_details[order_obj.order_id] = {
                            "sl_price": order_data["sl_price"], "tp_price": order_data["tp_price"],
                            "volume_filled": order_obj.volume_traded, "instrument_id": order_obj.instrument_id
                        }
                        logger.info(f"    Stored SL/TP for filled opening order {order_obj.order_id} for {sym}.")
                    del active_orders[sym] # Remove from active tracking once finished

            # --- 2. Update Current Positions ---
            for sym in INSTRUMENTS:
                pos = api.get_position(sym) # This object is updated in-place by api.wait_update()
                current_pos_details = current_positions.get(sym)
                precision = get_price_precision(get_tick_size(api, sym)) # For logging
                
                if pos and (pos.pos_long > 0 or pos.pos_short > 0): # Position exists
                    direction = "LONG" if pos.pos_long > 0 else "SHORT"
                    volume = pos.pos_long or pos.pos_short
                    # TQSDK's open_price_long/short is average open price for the position
                    open_price = pos.open_price_long if direction == "LONG" else pos.open_price_short 

                    if not current_pos_details: # New position detected
                        sl, tp, order_id_open = None, None, None
                        # Check if this new position resulted from one of our tracked opening orders
                        for oid, details in list(filled_opening_order_details.items()):
                            # Match by instrument and if filled volume matches current position volume
                            if details["instrument_id"] == sym and details["volume_filled"] == volume : 
                                sl, tp, order_id_open = details["sl_price"], details["tp_price"], oid
                                logger.info(f"  Position Update: New position for {sym} linked to filled order {oid}. Assigning SL: {sl:.{precision}f}, TP: {tp:.{precision}f}")
                                del filled_opening_order_details[oid] # Consume the detail
                                break
                        
                        current_positions[sym] = {"direction": direction, "volume": volume, "open_price": open_price, 
                                                "sl_price": sl, "tp_price": tp, 
                                                "update_time": api.get_time(), "order_id_open": order_id_open}
                        if not order_id_open: 
                            logger.info(f"  Position Update: NEW {direction} position for {sym}: {volume} @ {open_price:.{precision}f}. SL/TP not assigned (no matching script order found).")
                        else:
                             logger.info(f"  Position Update: NEW {direction} position for {sym}: {volume} @ {open_price:.{precision}f} from order {order_id_open}. SL: {sl:.{precision}f if sl else 'N/A'}, TP: {tp:.{precision}f if tp else 'N/A'}")
                    
                    elif current_pos_details["volume"] != volume or current_pos_details["direction"] != direction: # Position changed
                        # This could be due to partial fills of a new order, or external position changes.
                        # SL/TP from an original opening order should ideally be preserved if it's the same logical position.
                        current_positions[sym].update({"direction": direction, "volume": volume, "open_price": open_price, "update_time": api.get_time()})
                        logger.info(f"  Position Update: MODIFIED {direction} position for {sym}: {volume} @ {open_price:.{precision}f}. SL/TP remain {current_pos_details.get('sl_price')}/{current_pos_details.get('tp_price')}.")
                    
                    # If position was already there (e.g. from initial load or previous cycle) but SL/TP were None, try to assign them now if a relevant order just filled.
                    elif current_pos_details and current_pos_details.get("sl_price") is None: 
                        order_id_open_to_check = current_pos_details.get("order_id_open") # This might be None if pre-existing
                        sl_tp_detail_source = filled_opening_order_details.get(order_id_open_to_check) if order_id_open_to_check else None
                        
                        if not sl_tp_detail_source: # Broader search if no specific order_id was linked or found
                            for oid, details in list(filled_opening_order_details.items()):
                                 # Match if instrument is same and new filled volume matches current position volume
                                 if details["instrument_id"] == sym and details["volume_filled"] == volume:
                                    sl_tp_detail_source = details
                                    order_id_to_check = oid
                                    current_positions[sym]["order_id_open"] = oid # Link it now
                                    logger.info(f"  Position Update: Linking existing position {sym} to order {oid} for SL/TP assignment.")
                                    del filled_opening_order_details[oid] # Consume
                                    break
                        
                        if sl_tp_detail_source:
                            current_positions[sym]["sl_price"] = sl_tp_detail_source["sl_price"]
                            current_positions[sym]["tp_price"] = sl_tp_detail_source["tp_price"]
                            logger.info(f"  Position Update: Assigned SL/TP to position {sym} from order {order_id_to_check}. SL: {sl_tp_detail_source['sl_price']:.{precision}f}, TP: {sl_tp_detail_source['tp_price']:.{precision}f}")
                
                elif current_pos_details: # Position is now flat (pos.pos_long/short are 0 or pos is None)
                    logger.info(f"  Position Update: Position for {sym} (was {current_pos_details['direction']} {current_pos_details['volume']}) is now FLAT.")
                    del current_positions[sym]
            
            # --- 3. Risk Management: Check SL/TP for active positions ---
            for sym, pos_details in list(current_positions.items()): # Iterate copy
                if pos_details.get("sl_price") is None or pos_details.get("tp_price") is None:
                    # This position does not have SL/TP set by the script (e.g., pre-existing or SL/TP transfer failed)
                    continue 
                
                # Check if a closing order for this symbol is already active
                if sym in active_orders and active_orders[sym]["purpose"] == ORDER_PURPOSE_CLOSE:
                    continue # Already attempting to close

                quote = api.get_quote(sym) # Quote should be subscribed
                if not quote or pd.isna(quote.last_price): 
                    logger.warning(f"  Risk Mgt: No valid quote for {sym} to check SL/TP.")
                    continue

                current_price = quote.last_price
                should_close, close_reason = False, ""
                sl_p, tp_p = pos_details["sl_price"], pos_details["tp_price"]
                precision = get_price_precision(get_tick_size(api, sym)) # For logging

                if pos_details["direction"] == "LONG":
                    if current_price <= sl_p: should_close, close_reason = True, f"Stop-loss (<= {sl_p:.{precision}f})"
                    elif current_price >= tp_p: should_close, close_reason = True, f"Take-profit (>= {tp_p:.{precision}f})"
                elif pos_details["direction"] == "SHORT":
                    if current_price >= sl_p: should_close, close_reason = True, f"Stop-loss (>= {sl_p:.{precision}f})"
                    elif current_price <= tp_p: should_close, close_reason = True, f"Take-profit (<= {tp_p:.{precision}f})"
                
                if should_close:
                    logger.info(f"  Risk Mgt: Closing {sym} ({pos_details['direction']} {pos_details['volume']}) due to: {close_reason}. Last Price: {current_price:.{precision}f}")
                    closing_order_obj = close_position(api, sym, pos_details)
                    if closing_order_obj: 
                        active_orders[sym] = {"order": closing_order_obj, "sl_price": None, "tp_price": None, "purpose": ORDER_PURPOSE_CLOSE}
                    else: 
                        logger.warning(f"  Risk Mgt: Failed to place closing order for {sym}. Will retry on next cycle.")
    
    finally: # Cleanup within run_bot
        if api and not api.is_closed(): # Ensure api is valid before using
            logger.info(f"Initiating shutdown process within run_bot at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
            for sym, order_data in list(active_orders.items()):
                order = order_data.get("order")
                try:
                    if order and hasattr(order, 'order_id') and hasattr(order, 'status') and \
                       order.status == "ALIVE" and order_data.get("purpose") == ORDER_PURPOSE_OPEN :
                         logger.info(f"  Cancelling ALIVE opening order {order.order_id} for {sym}...")
                         api.cancel_order(order)
                except Exception as cancel_e: 
                    logger.error(f"  Error cancelling order for {sym} (ID: {order.order_id if order else 'N/A'}): {cancel_e}", exc_info=False)


if __name__ == "__main__":
    # --- Logger Configuration ---
    logger.setLevel(logging.INFO) 
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    try:
        file_handler = logging.FileHandler('gap_trader.log', mode='a') 
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"Failed to set up file handler for logging: {e}", exc_info=False)
    # --- End Logger Configuration ---

    api_instance = None 
    try:
        logger.info("Initializing TQApi...")
        api_instance = TqApi(auth=TqAuth())
        logger.info(f"TQApi initialized. TQSDK Version: {api_instance.lib_version}, Broker: {api_instance.broker_id if api_instance.broker_id else 'N/A'}")
        account = api_instance.get_account()
        if account: logger.info(f"Account ID: {account.account_id}, Balance: {account.balance}, Available: {account.available}")
        
        run_bot(api_instance) 
    except KeyboardInterrupt: 
        logger.info("\nTrading bot stopped by user (KeyboardInterrupt).")
    except Exception as e: 
        logger.exception(f"A critical error occurred in the main execution: {e}") 
    finally:
        if api_instance and not api_instance.is_closed(): 
            logger.info(f"Closing API connection from __main__ at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
            api_instance.close() 
            logger.info("API connection closed.")
        else: 
            logger.info("API was not initialized or already closed.")
    logger.info("Trading bot script finished.")
