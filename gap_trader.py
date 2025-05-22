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
    Fetches the tick size for a given instrument symbol from TQSDK.
    Includes a fallback heuristic if the quote or tick_size is unavailable.
    Args:
        api: TqApi instance.
        instrument_symbol: The symbol of the instrument.
    Returns:
        The tick size for the instrument.
    """
    quote = api.get_quote(instrument_symbol)
    if quote and not pd.isna(quote.tick_size) and quote.tick_size > 0:
        return quote.tick_size
    
    logger.warning(f"Could not get valid tick_size for {instrument_symbol} from quote. Using fallback heuristic.")
    if "SHFE.au" in instrument_symbol or "INE.au" in instrument_symbol : return 0.01 
    if "SHFE.ag" in instrument_symbol : return 1.0
    if "SHFE.rb" in instrument_symbol or "SHFE.hc" in instrument_symbol : return 1.0
    if "DCE.i" in instrument_symbol : return 0.5
    if "CZCE.MA" in instrument_symbol: return 1.0
    if "INE.sc" in instrument_symbol: return 0.1
    logger.warning(f"No specific fallback tick size for {instrument_symbol}, defaulting to 1.0. Price rounding may be inaccurate.")
    return 1.0 

def get_price_precision(tick_size: float) -> int:
    """
    Determines the number of decimal places for price precision based on tick size.
    Args:
        tick_size: The tick size of the instrument.
    Returns:
        The number of decimal places for price precision.
    """
    if tick_size == 0: return 2 
    s = f"{tick_size:.10f}" 
    if '.' in s:
        return len(s.split('.')[1].rstrip('0')) 
    return 0 

def round_to_tick(price: float, tick_size_val: float) -> float:
    """
    Rounds a given price to the nearest valid tick based on the instrument's tick size.
    Args:
        price: The price to round.
        tick_size_val: The tick size of the instrument.
    Returns:
        The price rounded to the nearest valid tick.
    """
    if tick_size_val == 0: return price 
    precision = get_price_precision(tick_size_val)
    return round(round(price / tick_size_val) * tick_size_val, precision)

# --- Core Trading Logic Functions ---
def check_opening_gap(api: TqApi, instrument_symbol: str) -> dict | None:
    """
    Checks for an opening gap for the specified instrument.
    Args:
        api: TqApi instance.
        instrument_symbol: The symbol of the instrument.
    Returns:
        A dictionary with gap details or an error dictionary.
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
    Determines a trade signal based on gap information, calculating SL/TP prices.
    Args:
        api: TqApi instance (for fetching tick_size).
        gap_info: Dictionary from check_opening_gap.
    Returns:
        A dictionary with trade signal details or None.
    """
    if not gap_info or gap_info.get("error") or not isinstance(gap_info.get("current_open"), (int, float)): 
        return None
    
    instrument = gap_info["instrument"]
    gap_type = gap_info["gap_type"]
    current_open = gap_info["current_open"]
    gap_percentage = gap_info.get("gap_percentage", 0.0)
    
    tick_size = get_tick_size(api, instrument)
    entry_price = current_open 

    if gap_type == "up-gap": 
        signal_type = "SELL_SHORT"
        stop_loss = round_to_tick(entry_price * (1 + STOP_LOSS_PERCENTAGE / 100), tick_size)
        take_profit = round_to_tick(entry_price * (1 - TAKE_PROFIT_PERCENTAGE / 100), tick_size)
    elif gap_type == "down-gap": 
        signal_type = "BUY_OPEN"
        stop_loss = round_to_tick(entry_price * (1 - STOP_LOSS_PERCENTAGE / 100), tick_size)
        take_profit = round_to_tick(entry_price * (1 + TAKE_PROFIT_PERCENTAGE / 100), tick_size)
    else: 
        return None
        
    return {"instrument": instrument, "signal_type": signal_type, "entry_price_target": entry_price, 
            "stop_loss_price": stop_loss, "take_profit_price": take_profit, 
            "reason": f"{gap_type.replace('-', ' ').title()} of {gap_percentage:.2f}%"}

def execute_trade(api: TqApi, trade_signal: dict | None) -> TqApi.Order | None:
    """
    Places a LIMIT order based on the trade_signal.
    Args:
        api: TqApi instance.
        trade_signal: Dictionary from determine_trade_signal.
    Returns:
        TQSDK Order object if successful, otherwise None.
    """
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
    Closes an existing position using a MARKET order.
    Args:
        api: TqApi instance.
        instrument_symbol: Symbol of the position to close.
        pos_details: Dictionary containing position details (direction, volume).
    Returns:
        TQSDK Order object if successful, otherwise None.
    """
    direction_to_close = "SELL" if pos_details["direction"] == "LONG" else "BUY"
    volume_to_close = pos_details["volume"]
    logger.info(f"Close Position: Attempting {direction_to_close} CLOSE {volume_to_close} of {instrument_symbol} (MARKET).")
    try:
        closing_order = api.insert_order(symbol=instrument_symbol, direction=direction_to_close, 
                                         offset="CLOSE", volume=volume_to_close, order_type="MARKET")
        return closing_order
    except Exception as e: 
        logger.error(f"Close Position Error for {instrument_symbol}: {e}", exc_info=False)
        return None

# --- Main Application Logic ---
def run_bot(api: TqApi):
    """
    Main function to run the trading bot's logic.
    Handles initialization, main loop (order processing, position updates, risk management),
    and ensures proper state management.
    Args:
        api: Initialized TqApi instance.
    """
    active_orders = {}     
    current_positions = {} 
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
                                        "order_id_open": None} 
                precision = get_price_precision(get_tick_size(api, sym))
                logger.info(f"  Initial Position: {direction} {volume} of {sym} at {open_price:.{precision}f} (SL/TP not script-defined).")
        logger.info("Initial position check complete.\n" + "="*50)

        logger.info("Analyzing gaps for initial trades...")
        for sym in INSTRUMENTS:
            logger.info(f"  Processing for Gap Analysis: {sym}")
            gap_info = check_opening_gap(api, sym)
            precision = get_price_precision(get_tick_size(api, sym))
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
            logger.info("-"*40) 
        logger.info("Initial gap analysis and trade placement complete.\n" + "="*50)

        logger.info(f"Bot monitoring loop started: {api.get_time().strftime('%Y-%m-%d %H:%M:%S')}")
        while True:
            api.wait_update(deadline=api.get_time() + pd.Timedelta(seconds=1))

            # --- 1. Process Active Orders ---
            for sym, order_data in list(active_orders.items()):
                order_obj = order_data["order"]
                if order_obj.is_finished():
                    logger.info(f"  Order Monitor: Order for {sym} (ID: {order_obj.order_id}, Purpose: {order_data['purpose']}) FINISHED. Status: {order_obj.status_message}, Vol Traded: {order_obj.volume_traded}/{order_obj.volume_orign}")
                    if order_data["purpose"] == ORDER_PURPOSE_OPEN and order_obj.volume_traded > 0:
                        filled_opening_order_details[order_obj.order_id] = {
                            "sl_price": order_data["sl_price"], "tp_price": order_data["tp_price"],
                            "volume_filled": order_obj.volume_traded, "instrument_id": order_obj.instrument_id
                        }
                        logger.info(f"    Stored SL/TP for filled opening order {order_obj.order_id} for {sym}.")
                    del active_orders[sym]

            # --- 2. Update Current Positions ---
            for sym in INSTRUMENTS:
                pos = api.get_position(sym)
                current_pos_details = current_positions.get(sym)
                precision = get_price_precision(get_tick_size(api, sym))
                
                if pos and (pos.pos_long > 0 or pos.pos_short > 0):
                    direction = "LONG" if pos.pos_long > 0 else "SHORT"
                    volume = pos.pos_long or pos.pos_short
                    open_price = pos.open_price_long or pos.open_price_short 

                    if not current_pos_details: 
                        sl, tp, order_id_open = None, None, None
                        for oid, details in list(filled_opening_order_details.items()):
                            if details["instrument_id"] == sym and details["volume_filled"] == volume : 
                                sl, tp, order_id_open = details["sl_price"], details["tp_price"], oid
                                logger.info(f"  Position Update: New position for {sym} linked to filled order {oid}. Assigning SL: {sl:.{precision}f}, TP: {tp:.{precision}f}")
                                del filled_opening_order_details[oid]
                                break
                        current_positions[sym] = {"direction": direction, "volume": volume, "open_price": open_price, 
                                                "sl_price": sl, "tp_price": tp, 
                                                "update_time": api.get_time(), "order_id_open": order_id_open}
                        if not order_id_open: 
                            logger.info(f"  Position Update: NEW {direction} position for {sym}: {volume} @ {open_price:.{precision}f}. SL/TP not assigned (no matching script order found).")
                        else:
                            logger.info(f"  Position Update: NEW {direction} position for {sym}: {volume} @ {open_price:.{precision}f} from order {order_id_open}. SL: {sl:.{precision}f if sl else 'N/A'}, TP: {tp:.{precision}f if tp else 'N/A'}")
                    
                    elif current_pos_details["volume"] != volume or current_pos_details["direction"] != direction: 
                        current_positions[sym].update({"direction": direction, "volume": volume, "open_price": open_price, "update_time": api.get_time()})
                        logger.info(f"  Position Update: MODIFIED {direction} position for {sym}: {volume} @ {open_price:.{precision}f}. SL/TP remain {current_pos_details.get('sl_price')}/{current_pos_details.get('tp_price')}.")
                    
                    elif current_pos_details and current_pos_details.get("sl_price") is None: 
                        order_id_open_to_check = current_pos_details.get("order_id_open") 
                        sl_tp_detail_source = filled_opening_order_details.get(order_id_open_to_check) if order_id_open_to_check else None
                        if not sl_tp_detail_source: 
                            for oid, details in list(filled_opening_order_details.items()):
                                if details["instrument_id"] == sym and details["volume_filled"] == volume:
                                    sl_tp_detail_source = details; order_id_to_check = oid; current_positions[sym]["order_id_open"] = oid
                                    logger.info(f"  Position Update: Linking existing position {sym} to order {oid} for SL/TP.")
                                    del filled_opening_order_details[oid]; break
                        if sl_tp_detail_source:
                            current_positions[sym]["sl_price"] = sl_tp_detail_source["sl_price"]
                            current_positions[sym]["tp_price"] = sl_tp_detail_source["tp_price"]
                            logger.info(f"  Position Update: Assigned SL/TP to position {sym} from order {order_id_to_check}. SL: {sl_tp_detail_source['sl_price']:.{precision}f}, TP: {sl_tp_detail_source['tp_price']:.{precision}f}")
                
                elif current_pos_details: 
                    logger.info(f"  Position Update: Position for {sym} (was {current_pos_details['direction']} {current_pos_details['volume']}) is now FLAT.")
                    del current_positions[sym]
            
            # --- 3. Risk Management: Check SL/TP for active positions ---
            for sym, pos_details in list(current_positions.items()): 
                if pos_details.get("sl_price") is None or pos_details.get("tp_price") is None: continue
                if sym in active_orders and active_orders[sym]["purpose"] == ORDER_PURPOSE_CLOSE: continue

                quote = api.get_quote(sym)
                if not quote or pd.isna(quote.last_price): 
                    logger.warning(f"  Risk Mgt: No valid quote for {sym} to check SL/TP.")
                    continue

                current_price = quote.last_price
                should_close, close_reason = False, ""
                sl_p, tp_p = pos_details["sl_price"], pos_details["tp_price"]
                precision = get_price_precision(get_tick_size(api, sym))

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
    
    # Moved cleanup into run_bot's finally block for better encapsulation
    finally:
        if api and not api.is_closed():
            logger.info(f"Initiating shutdown process within run_bot at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
            # Cancel any ALIVE opening orders before closing
            for sym, order_data in list(active_orders.items()):
                order = order_data.get("order")
                try:
                    if order and hasattr(order, 'order_id') and hasattr(order, 'status') and \
                       order.status == "ALIVE" and order_data.get("purpose") == ORDER_PURPOSE_OPEN :
                         logger.info(f"  Cancelling ALIVE opening order {order.order_id} for {sym}...")
                         api.cancel_order(order)
                except Exception as cancel_e: 
                    logger.error(f"  Error cancelling order for {sym} (ID: {order.order_id if order else 'N/A'}): {cancel_e}", exc_info=False)
            # Note: Market closing orders (purpose=ORDER_PURPOSE_CLOSE) are typically not cancelled as they are expected to fill.
            # If any are still ALIVE, it might indicate an issue, but auto-cancellation might not be desired.
            # api.close() will be called by the main __name__ block's finally.


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
        api_instance = TqApi(auth=TqAuth())
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
