"""Earnings calendar and estimate tools."""

from __future__ import annotations

from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def get_earnings_calendar(ticker: str) -> str:
    """Get next earnings date, time (BMO/AMC), and estimated EPS for a ticker."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        cal = t.calendar

        lines = [f"=== {ticker.upper()} Earnings Calendar ==="]

        if cal is None:
            lines.append("No earnings calendar data available.")
            return "\n".join(lines)

        # calendar can be a dict or DataFrame depending on yfinance version
        if hasattr(cal, "to_dict"):
            # DataFrame style — convert to dict
            cal_dict = {col: cal[col].iloc[0] for col in cal.columns}
        elif isinstance(cal, dict):
            cal_dict = cal
        else:
            lines.append("Unrecognized calendar format.")
            return "\n".join(lines)

        earnings_date = cal_dict.get("Earnings Date", cal_dict.get("earningsDate", "N/A"))
        eps_avg = cal_dict.get("EPS Estimate", cal_dict.get("epsEstimate", "N/A"))
        revenue_avg = cal_dict.get("Revenue Estimate", cal_dict.get("revenueEstimate", "N/A"))
        earnings_time = cal_dict.get("Earnings Call Time", "N/A")

        # Determine BMO/AMC
        time_str = str(earnings_time).lower()
        if "before" in time_str or "bmo" in time_str:
            timing = "BMO (Before Market Open)"
        elif "after" in time_str or "amc" in time_str:
            timing = "AMC (After Market Close)"
        elif earnings_time != "N/A":
            timing = str(earnings_time)
        else:
            timing = "N/A"

        lines.append(f"Next Earnings Date:  {earnings_date}")
        lines.append(f"Earnings Time:       {timing}")
        lines.append(f"EPS Estimate:        {eps_avg}")
        if revenue_avg != "N/A":
            try:
                rev = float(revenue_avg)
                if rev >= 1e9:
                    rev_str = f"${rev / 1e9:.2f}B"
                elif rev >= 1e6:
                    rev_str = f"${rev / 1e6:.2f}M"
                else:
                    rev_str = f"${rev:.2f}"
                lines.append(f"Revenue Estimate:    {rev_str}")
            except (TypeError, ValueError):
                lines.append(f"Revenue Estimate:    {revenue_avg}")

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_earnings_calendar failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_earnings_estimate(ticker: str) -> str:
    """Get EPS estimates and trend for a ticker including current quarter, next quarter, and annual estimates."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        lines = [f"=== {ticker.upper()} Earnings Estimates ==="]
        found_data = False

        # Try earnings_estimate
        try:
            ee = t.earnings_estimate
            if ee is not None and not ee.empty:
                lines.append("\nEPS Estimates:")
                lines.append(ee.to_string())
                found_data = True
        except Exception:
            pass

        # Try eps_trend
        try:
            et = t.eps_trend
            if et is not None and not et.empty:
                lines.append("\nEPS Trend:")
                lines.append(et.to_string())
                found_data = True
        except Exception:
            pass

        # Try earnings_history as fallback
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty:
                lines.append("\nEarnings History (recent):")
                lines.append(eh.tail(4).to_string())
                found_data = True
        except Exception:
            pass

        # Try info fields
        try:
            info = t.info or {}
            eps_forward = info.get("forwardEps")
            eps_trailing = info.get("trailingEps")
            pe_forward = info.get("forwardPE")
            if eps_forward or eps_trailing:
                lines.append(f"\nForward EPS: {eps_forward}" if eps_forward else "")
                lines.append(f"Trailing EPS: {eps_trailing}" if eps_trailing else "")
                lines.append(f"Forward P/E: {pe_forward:.1f}" if pe_forward else "")
                found_data = True
        except Exception:
            pass

        if not found_data:
            lines.append("No earnings estimate data available.")

        result = "\n".join(line for line in lines if line)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_earnings_estimate failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"
