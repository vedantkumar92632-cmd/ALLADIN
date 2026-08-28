"""
Test TrailingATRStop functionality.
"""
import sys
sys.path.append(r'c:\Users\vedan\OneDrive\Desktop\artha')

from risk.pre_trade import TrailingATRStop

def test_trailing_stop():
    print("=" * 60)
    print("TESTING TRAILING ATR STOP")
    print("=" * 60)
    
    stop_mgr = TrailingATRStop(multiplier=2.0)
    ticker = "RELIANCE.NS"
    atr = 10.0
    
    # 1. Entry price 1000 -> Initial stop = 1000 - (2 * 10) = 980
    stop1, is_stopped = stop_mgr.update_and_check(ticker, 1000.0, atr, "LONG")
    print(f"Price 1000: Stop={stop1}, Stopped={is_stopped}")
    assert stop1 == 980.0
    assert not is_stopped
    
    # 2. Price rises to 1050 -> Stop moves up to 1050 - 20 = 1030
    stop2, is_stopped = stop_mgr.update_and_check(ticker, 1050.0, atr, "LONG")
    print(f"Price 1050: Stop={stop2}, Stopped={is_stopped}")
    assert stop2 == 1030.0
    assert not is_stopped
    
    # 3. Price drops to 1040 -> Stop STAYS at 1030 (never moves back)
    stop3, is_stopped = stop_mgr.update_and_check(ticker, 1040.0, atr, "LONG")
    print(f"Price 1040: Stop={stop3}, Stopped={is_stopped}")
    assert stop3 == 1030.0
    assert not is_stopped
    
    # 4. Price drops to 1025 -> Stopped out! (1025 <= 1030)
    stop4, is_stopped = stop_mgr.update_and_check(ticker, 1025.0, atr, "LONG")
    print(f"Price 1025: Stop={stop4}, Stopped={is_stopped}")
    assert is_stopped
    
    print("\n[OK] PASS: TrailingATRStop correctly trails peak price and triggers stop-out.")

if __name__ == "__main__":
    test_trailing_stop()
