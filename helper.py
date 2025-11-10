from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime

def inputvalidator(input_="ohlc"):
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1

            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }

            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]

            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(
                        'Must have a dataframe column named "{0}"'.format(inputs[l])
                    )

            return func(*args, **kwargs)

        return wrap

    return dfcheck


def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))

        return cls

    return decorate


@apply(inputvalidator(input_="ohlc"))
class candleComponent:
    __version__ = "0.0.26"

    @classmethod
    def fvg(cls, ohlc: DataFrame, join_consecutive=False) -> Series:
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.

        parameters:
        join_consecutive: bool - if there are multiple FVG in a row then they will be merged into one using the highest top and the lowest bottom

        returns:
        FVG = 1 if bullish fair value gap, -1 if bearish fair value gap
        Top = the top of the fair value gap
        Bottom = the bottom of the fair value gap
        MitigatedIndex = the index of the candle that mitigated the fair value gap
        """

        fvg = np.where(
            (
                (ohlc["high"].shift(1) < ohlc["low"].shift(-1))
                & (ohlc["close"] > ohlc["open"])
            )
            | (
                (ohlc["low"].shift(1) > ohlc["high"].shift(-1))
                & (ohlc["close"] < ohlc["open"])
            ),
            np.where(ohlc["close"] > ohlc["open"], 1, -1),
            np.nan,
        )

        top = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["low"].shift(-1),
                ohlc["low"].shift(1),
            ),
            np.nan,
        )

        bottom = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["high"].shift(1),
                ohlc["high"].shift(-1),
            ),
            np.nan,
        )

        # if there are multiple consecutive fvg then join them together using the highest top and lowest bottom and the last index
        if join_consecutive:
            for i in range(len(fvg) - 1):
                if fvg[i] == fvg[i + 1]:
                    top[i + 1] = max(top[i], top[i + 1])
                    bottom[i + 1] = min(bottom[i], bottom[i + 1])
                    fvg[i] = top[i] = bottom[i] = np.nan

        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if fvg[i] == 1:
                mask = ohlc["low"][i + 2 :] <= top[i]
            elif fvg[i] == -1:
                mask = ohlc["high"][i + 2 :] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        return pd.concat(
            [
                pd.Series(fvg, name="FVG"),
                pd.Series(top, name="Top"),
                pd.Series(bottom, name="Bottom"),
                pd.Series(mitigated_index, name="MitigatedIndex"),
            ],
            axis=1,
        )

    @classmethod
    def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50) -> Series:
        """
        Swing Highs and Lows
        A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
        A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

        parameters:
        swing_length: int - the amount of candles to look back and forward to determine the swing high or low

        returns:
        HighLow = 1 if swing high, -1 if swing low
        Level = the level of the swing high or low
        """

        swing_length *= 2
        # set the highs to 1 if the current high is the highest high in the last 5 candles and next 5 candles
        swing_highs_lows = np.where(
            ohlc["high"]
            == ohlc["high"].shift(-(swing_length // 2)).rolling(swing_length).max(),
            1,
            np.where(
                ohlc["low"]
                == ohlc["low"].shift(-(swing_length // 2)).rolling(swing_length).min(),
                -1,
                np.nan,
            ),
        )

        while True:
            positions = np.where(~np.isnan(swing_highs_lows))[0]

            if len(positions) < 2:
                break

            current = swing_highs_lows[positions[:-1]]
            next = swing_highs_lows[positions[1:]]

            highs = ohlc["high"].iloc[positions[:-1]].values
            lows = ohlc["low"].iloc[positions[:-1]].values

            next_highs = ohlc["high"].iloc[positions[1:]].values
            next_lows = ohlc["low"].iloc[positions[1:]].values

            index_to_remove = np.zeros(len(positions), dtype=bool)

            consecutive_highs = (current == 1) & (next == 1)
            index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
            index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)

            consecutive_lows = (current == -1) & (next == -1)
            index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
            index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)

            if not index_to_remove.any():
                break

            swing_highs_lows[positions[index_to_remove]] = np.nan

        positions = np.where(~np.isnan(swing_highs_lows))[0]

        if len(positions) > 0:
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            if swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1
            if swing_highs_lows[positions[-1]] == -1:
                swing_highs_lows[-1] = 1
            if swing_highs_lows[positions[-1]] == 1:
                swing_highs_lows[-1] = -1

        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )

        return pd.concat(
            [
                pd.Series(swing_highs_lows, name="HighLow"),
                pd.Series(level, name="Level"),
            ],
            axis=1,
        )
    
    @classmethod
    def bos_choch(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, close_break: bool = True
    ) -> Series:
        """
        BOS - Break of Structure
        CHoCH - Change of Character
        these are both indications of market structure changing

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
        close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.

        returns:
        BOS = 1 if bullish break of structure, -1 if bearish break of structure
        CHOCH = 1 if bullish change of character, -1 if bearish change of character
        Level = the level of the break of structure or change of character
        BrokenIndex = the index of the candle that broke the level
        """

        swing_highs_lows = swing_highs_lows.copy()

        level_order = []
        highs_lows_order = []

        bos = np.zeros(len(ohlc), dtype=np.int32)
        choch = np.zeros(len(ohlc), dtype=np.int32)
        level = np.zeros(len(ohlc), dtype=np.float32)

        last_positions = []

        for i in range(len(swing_highs_lows["HighLow"])):
            if not np.isnan(swing_highs_lows["HighLow"][i]):
                level_order.append(swing_highs_lows["Level"][i])
                highs_lows_order.append(swing_highs_lows["HighLow"][i])
                if len(level_order) >= 4:
                    # bullish bos
                    bos[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-4]
                                < level_order[-2]
                                < level_order[-3]
                                < level_order[-1]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bearish bos
                    bos[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-4]
                                > level_order[-2]
                                > level_order[-3]
                                > level_order[-1]
                            )
                        )
                        else bos[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bullish choch
                    choch[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-1]
                                > level_order[-3]
                                > level_order[-4]
                                > level_order[-2]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                    # bearish choch
                    choch[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-1]
                                < level_order[-3]
                                < level_order[-4]
                                < level_order[-2]
                            )
                        )
                        else choch[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                last_positions.append(i)

        broken = np.zeros(len(ohlc), dtype=np.int32)
        trend_bos = np.zeros(len(ohlc), dtype=np.int32)
        trend_choch = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            # if the bos is 1 then check if the candles high has gone above the level
            if bos[i] == 1 or choch[i] == 1:
                mask = ohlc["close" if close_break else "high"][i + 2 :] > level[i]
            # if the bos is -1 then check if the candles low has gone below the level
            elif bos[i] == -1 or choch[i] == -1:
                mask = ohlc["close" if close_break else "low"][i + 2 :] < level[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                broken[i] = j
                # determine trend direction
                # Mark trend at breakout candle (BrokenIndex)
                if bos[i] != 0:
                    trend_bos[j] = 1 if bos[i] == 1 else -1
                if choch[i] != 0:
                    trend_choch[j] = 1 if choch[i] == 1 else -1
                # if there are any unbroken bos or choch that started before this one and ended after this one then remove them
                for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                    if k < i and broken[k] >= j:
                        bos[k] = 0
                        choch[k] = 0
                        level[k] = 0
                        

        # remove the ones that aren't broken
        for i in np.where(
            np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
        )[0]:
            bos[i] = 0
            choch[i] = 0
            level[i] = 0
          
            

        # replace all the 0s with np.nan
        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        level = np.where(level != 0, level, np.nan)
        broken = np.where(broken != 0, broken, np.nan)

        bos = pd.Series(bos, name="BOS")
        choch = pd.Series(choch, name="CHOCH")
        level = pd.Series(level, name="Level")
        broken = pd.Series(broken, name="BrokenIndex")
        trend_bos = pd.Series(np.where(trend_bos != 0, trend_bos, np.nan), name="Trend_BOS")
        trend_choch = pd.Series(np.where(trend_choch != 0, trend_choch, np.nan), name="Trend_CHOCH")

        # ✅ Persist the latest trend value forward
        trend_bos = trend_bos.ffill()
        trend_choch = trend_choch.ffill()


        return pd.concat([bos, choch, level, broken,trend_bos, trend_choch], axis=1)

    @classmethod
    def ob(
    cls,
    ohlc: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
    close_mitigation: bool = False,
) -> pd.DataFrame:
        """
        Order Block (OB) detection with separate Bullish/Bearish columns.
        Propagates OB info (Top, Bottom, Created, Mitigated, OBVolume, Percentage)
        to all candles where the OB is active (between creation and mitigation).

        Parameters
        ----------
        ohlc : pd.DataFrame
            Must contain ['open','high','low','close','volume']
        swing_highs_lows : pd.DataFrame
            Must contain ['HighLow'] with 1 for swing high and -1 for swing low
        close_mitigation : bool
            If True, mitigation uses candle close; otherwise high/low.

        Returns
        -------
        pd.DataFrame
            Columns:
            - BullishTop, BullishBottom, BullishCreated, BullishMitigated, BullishActive
            - BearishTop, BearishBottom, BearishCreated, BearishMitigated, BearishActive
            - OBVolume, Percentage
        """

        n = len(ohlc)
        _open, _high, _low, _close, _volume = (
            ohlc["open"].values,
            ohlc["high"].values,
            ohlc["low"].values,
            ohlc["close"].values,
            ohlc["volume"].values,
        )
        swing_hl = swing_highs_lows["HighLow"].values

        # Preallocate
        crossed = np.full(n, False, dtype=bool)
        swing_high_idx = np.flatnonzero(swing_hl == 1)
        swing_low_idx = np.flatnonzero(swing_hl == -1)

        # Order block containers
        bullish_obs, bearish_obs = [], []

        # --- Detect Bullish OBs ---
        active_bullish = []
        for i in range(n):
            # Update mitigation
            for ob in active_bullish.copy():
                if ob["mitigated"] is None:
                    if ((not close_mitigation and _low[i] < ob["bottom"]) or
                        (close_mitigation and min(_open[i], _close[i]) < ob["bottom"])):
                        ob["mitigated"] = i
                        active_bullish.remove(ob)

            # New bullish OB
            pos = np.searchsorted(swing_high_idx, i)
            last_top = swing_high_idx[pos - 1] if pos > 0 else None

            if last_top is not None:
                if _close[i] > _high[last_top] and not crossed[last_top]:
                    crossed[last_top] = True
                    default_idx = i - 1
                    ob_btm, ob_top, ob_idx = _high[default_idx], _low[default_idx], default_idx

                    if i - last_top > 1:
                        start, end = last_top + 1, i
                        seg = _low[start:end]
                        if len(seg):
                            min_val = seg.min()
                            cands = np.nonzero(seg == min_val)[0]
                            if cands.size:
                                ci = start + cands[-1]
                                ob_btm, ob_top, ob_idx = _low[ci], _high[ci], ci

                    # Volume info
                    v0 = _volume[i]
                    v1 = _volume[i - 1] if i >= 1 else 0
                    v2 = _volume[i - 2] if i >= 2 else 0
                    vol_sum = v0 + v1 + v2
                    low_v, high_v = v2, v0 + v1
                    max_v = max(high_v, low_v)
                    pct = (min(high_v, low_v) / max_v * 100) if max_v != 0 else 100

                    ob_data = {
                        "type": "bullish",
                        "top": ob_top,
                        "bottom": ob_btm,
                        "created": i,
                        "mitigated": None,
                        "volume": vol_sum,
                        "percentage": pct,
                    }
                    bullish_obs.append(ob_data)
                    active_bullish.append(ob_data)

        # --- Detect Bearish OBs ---
        active_bearish = []
        for i in range(n):
            # Update mitigation
            for ob in active_bearish.copy():
                if ob["mitigated"] is None:
                    if ((not close_mitigation and _high[i] > ob["top"]) or
                        (close_mitigation and max(_open[i], _close[i]) > ob["top"])):
                        ob["mitigated"] = i
                        active_bearish.remove(ob)

            # New bearish OB
            pos = np.searchsorted(swing_low_idx, i)
            last_btm = swing_low_idx[pos - 1] if pos > 0 else None

            if last_btm is not None:
                if _close[i] < _low[last_btm] and not crossed[last_btm]:
                    crossed[last_btm] = True
                    default_idx = i - 1
                    ob_top, ob_btm, ob_idx = _high[default_idx], _low[default_idx], default_idx

                    if i - last_btm > 1:
                        start, end = last_btm + 1, i
                        seg = _high[start:end]
                        if len(seg):
                            max_val = seg.max()
                            cands = np.nonzero(seg == max_val)[0]
                            if cands.size:
                                ci = start + cands[-1]
                                ob_top, ob_btm, ob_idx = _high[ci], _low[ci], ci

                    v0 = _volume[i]
                    v1 = _volume[i - 1] if i >= 1 else 0
                    v2 = _volume[i - 2] if i >= 2 else 0
                    vol_sum = v0 + v1 + v2
                    low_v, high_v = v0 + v1, v2
                    max_v = max(high_v, low_v)
                    pct = (min(high_v, low_v) / max_v * 100) if max_v != 0 else 100

                    ob_data = {
                        "type": "bearish",
                        "top": ob_top,
                        "bottom": ob_btm,
                        "created": i,
                        "mitigated": None,
                        "volume": vol_sum,
                        "percentage": pct,
                    }
                    bearish_obs.append(ob_data)
                    active_bearish.append(ob_data)

        # === Fill Candle Data ===
        cols = {
            "BullishTop": np.full(n, np.nan),
            "BullishBottom": np.full(n, np.nan),
            "BullishCreated": np.full(n, np.nan),
            "BullishMitigated": np.full(n, np.nan),
            "BullishActive": np.zeros(n),
            "BullishVolume": np.full(n, np.nan),
            "BullishPercentage": np.full(n, np.nan),

            "BearishTop": np.full(n, np.nan),
            "BearishBottom": np.full(n, np.nan),
            "BearishCreated": np.full(n, np.nan),
            "BearishMitigated": np.full(n, np.nan),
            "BearishActive": np.zeros(n),
            "BearishVolume": np.full(n, np.nan),
            "BearishPercentage": np.full(n, np.nan),
        }

        # Fill bullish OB info
        for ob in bullish_obs:
            start = int(ob["created"])
            end = int(ob["mitigated"]) if ob["mitigated"] is not None else n
            cols["BullishTop"][start:end] = ob["top"]
            cols["BullishBottom"][start:end] = ob["bottom"]
            cols["BullishCreated"][start:end] = ob["created"]
            cols["BullishMitigated"][start:end] = ob["mitigated"] if ob["mitigated"] is not None else np.nan
            cols["BullishActive"][start:end] = 1
            cols["BullishVolume"][start:end] = ob["volume"]
            cols["BullishPercentage"][start:end] = ob["percentage"]

        # Fill bearish OB info
        for ob in bearish_obs:
            start = int(ob["created"])
            end = int(ob["mitigated"]) if ob["mitigated"] is not None else n
            cols["BearishTop"][start:end] = ob["top"]
            cols["BearishBottom"][start:end] = ob["bottom"]
            cols["BearishCreated"][start:end] = ob["created"]
            cols["BearishMitigated"][start:end] = ob["mitigated"] if ob["mitigated"] is not None else np.nan
            cols["BearishActive"][start:end] = 1
            cols["BearishVolume"][start:end] = ob["volume"]
            cols["BearishPercentage"][start:end] = ob["percentage"]

        return pd.DataFrame(cols)
            

    @classmethod
    def retracements(cls, ohlc: DataFrame, swing_highs_lows: DataFrame) -> Series:
        """
        Retracement
        This method returns the percentage of a retracement from the swing high or low

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function

        returns:
        Direction = 1 if bullish retracement, -1 if bearish retracement
        CurrentRetracement% = the current retracement percentage from the swing high or low
        DeepestRetracement% = the deepest retracement percentage from the swing high or low
        """

        swing_highs_lows = swing_highs_lows.copy()

        direction = np.zeros(len(ohlc), dtype=np.int32)
        current_retracement = np.zeros(len(ohlc), dtype=np.float64)
        deepest_retracement = np.zeros(len(ohlc), dtype=np.float64)

        top = 0
        bottom = 0
        for i in range(len(ohlc)):
            if swing_highs_lows["HighLow"][i] == 1:
                direction[i] = 1
                top = swing_highs_lows["Level"][i]
                # deepest_retracement[i] = 0
            elif swing_highs_lows["HighLow"][i] == -1:
                direction[i] = -1
                bottom = swing_highs_lows["Level"][i]
                # deepest_retracement[i] = 0
            else:
                direction[i] = direction[i - 1] if i > 0 else 0

            if direction[i - 1] == 1:
                current_retracement[i] = round(
                    100 - (((ohlc["low"].iloc[i] - bottom) / (top - bottom)) * 100), 1
                )
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[i - 1]
                        if i > 0 and direction[i - 1] == 1
                        else 0
                    ),
                    current_retracement[i],
                )
            if direction[i] == -1:
                current_retracement[i] = round(
                    100 - ((ohlc["high"].iloc[i] - top) / (bottom - top)) * 100, 1
                )
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[i - 1]
                        if i > 0 and direction[i - 1] == -1
                        else 0
                    ),
                    current_retracement[i],
                )

        # shift the arrays by 1
        current_retracement = np.roll(current_retracement, 1)
        deepest_retracement = np.roll(deepest_retracement, 1)
        direction = np.roll(direction, 1)

        # remove the first 3 retracements as they get calculated incorrectly due to not enough data
        remove_first_count = 0
        for i in range(len(direction)):
            if i + 1 == len(direction):
                break
            if direction[i] != direction[i + 1]:
                remove_first_count += 1
            direction[i] = 0
            current_retracement[i] = 0
            deepest_retracement[i] = 0
            if remove_first_count == 3:
                direction[i + 1] = 0
                current_retracement[i + 1] = 0
                deepest_retracement[i + 1] = 0
                break

        direction = pd.Series(direction, name="Direction")
        current_retracement = pd.Series(current_retracement, name="CurrentRetracement%")
        deepest_retracement = pd.Series(deepest_retracement, name="DeepestRetracement%")

        return pd.concat([direction, current_retracement, deepest_retracement], axis=1)
