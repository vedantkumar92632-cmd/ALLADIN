import datetime
from zoneinfo import ZoneInfo
from typing import Union, Set

# Timezone definitions
IST = ZoneInfo("Asia/Kolkata")

# Hardcoded NSE/BSE holidays for 2026
# These are the standard holidays observed by National Stock Exchange of India.
NSE_HOLIDAYS_2026: Set[datetime.date] = {
    datetime.date(2026, 1, 26),  # Republic Day
    datetime.date(2026, 3, 4),   # Holi
    datetime.date(2026, 3, 20),  # Eid-ul-Fitr (Ramzan Id)
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 4, 14),  # Dr. Ambedkar Jayanti
    datetime.date(2026, 5, 1),   # Maharashtra Day
    datetime.date(2026, 5, 27),  # Bakri Id (Eid-al-Adha)
    datetime.date(2026, 6, 25),  # Muharram
    datetime.date(2026, 8, 15),  # Independence Day (Saturday)
    datetime.date(2026, 10, 2),  # Mahatma Gandhi Jayanti
    datetime.date(2026, 10, 20), # Dussehra
    datetime.date(2026, 11, 9),  # Diwali (Laxmi Puja)
    datetime.date(2026, 11, 23), # Gurunanak Jayanti
    datetime.date(2026, 12, 25), # Christmas
}

# Standard NSE/BSE Market Hours (IST)
PRE_OPEN_START = datetime.time(9, 0, 0)
PRE_OPEN_END = datetime.time(9, 15, 0)
REGULAR_START = datetime.time(9, 15, 0)
REGULAR_END = datetime.time(15, 30, 0)

def get_now_ist() -> datetime.datetime:
    """Return the current time in the Asia/Kolkata timezone."""
    return datetime.datetime.now(IST)

def to_ist(dt: datetime.datetime) -> datetime.datetime:
    """Convert any naive or aware datetime to Asia/Kolkata timezone."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc).astimezone(IST)
    return dt.astimezone(IST)

def is_trading_day(dt: Union[datetime.datetime, datetime.date]) -> bool:
    """
    Check if a given date/datetime is a trading day for NSE/BSE.
    Excludes weekends and Indian market holidays.
    """
    date_val = dt.date() if isinstance(dt, datetime.datetime) else dt
    
    # Check weekend (Saturday = 5, Sunday = 6)
    if date_val.weekday() >= 5:
        return False
        
    # Check holiday list
    if date_val in NSE_HOLIDAYS_2026:
        return False
        
    return True

def is_market_open(dt: datetime.datetime, include_pre_open: bool = False) -> bool:
    """
    Check if the market is open at the given datetime.
    """
    dt_ist = to_ist(dt)
    if not is_trading_day(dt_ist):
        return False
        
    time_val = dt_ist.time()
    
    start_time = PRE_OPEN_START if include_pre_open else REGULAR_START
    return start_time <= time_val <= REGULAR_END

def get_next_market_open(dt: datetime.datetime) -> datetime.datetime:
    """
    Find the next market open datetime (regular session start at 9:15 AM IST)
    following the given datetime.
    """
    dt_ist = to_ist(dt)
    candidate_date = dt_ist.date()
    
    # If it is today and we are before regular start, candidate is today
    if is_trading_day(candidate_date) and dt_ist.time() < REGULAR_START:
        pass
    else:
        # Move to tomorrow and find next trading day
        candidate_date += datetime.timedelta(days=1)
        while not is_trading_day(candidate_date):
            candidate_date += datetime.timedelta(days=1)
            
    return datetime.datetime.combine(candidate_date, REGULAR_START, tzinfo=IST)

def get_next_market_close(dt: datetime.datetime) -> datetime.datetime:
    """
    Find the next market close datetime (regular session end at 3:30 PM IST)
    following the given datetime.
    """
    dt_ist = to_ist(dt)
    candidate_date = dt_ist.date()
    
    # If it is today and we are before regular end, candidate is today
    if is_trading_day(candidate_date) and dt_ist.time() < REGULAR_END:
        pass
    else:
        # Move to tomorrow and find next trading day
        candidate_date += datetime.timedelta(days=1)
        while not is_trading_day(candidate_date):
            candidate_date += datetime.timedelta(days=1)
            
    return datetime.datetime.combine(candidate_date, REGULAR_END, tzinfo=IST)
