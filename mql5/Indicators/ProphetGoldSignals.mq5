//+------------------------------------------------------------------+
//| ProphetGoldSignals.mq5                                           |
//| EUR/USD Prophet gold signals on the chart (openspec change       |
//| gold-mt5-chart-signals, spec gold-forecasting/mt5-chart-signals). |
//|                                                                  |
//| Reads Common\Files\EURUSDProphet\gold_signals.tsv, which the     |
//| gold logger rebuilds every hour (src/gold/mt5_export.py), and    |
//| draws on a GOLD / XAUUSD chart at H1 or lower:                   |
//|   - an up/down arrow on each hour the h1_lstm forecast;          |
//|   - a two-headed marker on each hour a large-move alert covers;  |
//|   - a short text and a hover description beside each mark;       |
//|   - a corner panel with the current state.                       |
//| Every word comes from the file except the status lines in T().   |
//| An indicator: it reads one file and draws, nothing else.         |
//+------------------------------------------------------------------+
#property copyright   "EUR/USD Prophet"
#property version     "1.00"
#property description "Gold research signals: h1_lstm direction and large-move alerts."
#property description "Research - no proven edge, gross of cost."
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

enum ENUM_PGS_LANG
  {
   PGS_BG = 0, // Български
   PGS_EN = 1  // English
  };

input ENUM_PGS_LANG    InpLanguage   = PGS_BG;             // Език / Language
input color            InpUpColor    = clrDodgerBlue;      // Up arrow colour
input color            InpDownColor  = clrOrangeRed;       // Down arrow colour
input color            InpAlertColor = clrGold;            // Large-move marker colour
input ENUM_BASE_CORNER InpCorner     = CORNER_RIGHT_UPPER; // Panel corner
input int              InpFontSize   = 8;                  // Font size

#define PGS_PREFIX     "PGS_"
#define PGS_FILE       "EURUSDProphet\\gold_signals.tsv"
#define PGS_VERSION    "1"
#define PGS_CODE_UP    233
#define PGS_CODE_DOWN  234
#define PGS_CODE_ALERT 244
#define PGS_STALE_S    7200
#define PGS_TRADING_S  600
#define PGS_STEP_PX    18
#define PGS_FONT       "Segoe UI"

struct PgsMark
  {
   string            kind;
   string            key;
   datetime          hour;       // bar-label clock (the logger's Europe/Berlin)
   datetime          hour_end;
   string            direction;
   string            outcome;
   double            close;
   string            text;
   string            tip;
   string            panel;
   string            hour_sofia;
  };

PgsMark  g_marks[];
bool     g_loaded = false;          // a file has been read at least once
int      g_state = 1;               // 0 ok, 1 missing or unreadable, 2 unknown format
datetime g_file_mod = 0;
datetime g_generated = 0;           // UTC
int      g_label_offset_min = 0;
string   g_written = "", g_count = "", g_research = "";
string   g_lang = "bg";
datetime g_last_bar = 0;

//+------------------------------------------------------------------+
//| The status lines the file cannot carry, BG and EN.               |
//+------------------------------------------------------------------+
string T(const string id)
  {
   bool bg = (InpLanguage == PGS_BG);
   if(id == "title")
      return bg ? "Злато · h1_lstm" : "Gold · h1_lstm";
   if(id == "not_gold")
      return bg ? "Това не е графика на злато — няма сигнали" : "This chart is not gold — no signals";
   if(id == "timeframe")
      return bg ? "Сигналите се показват на H1 или по-малък таймфрейм" : "Signals show on H1 or a lower timeframe";
   if(id == "missing")
      return bg ? "Няма данни за златото. Изпълни: python -m src.gold.mt5_export" : "No gold data found. Run: python -m src.gold.mt5_export";
   if(id == "unknown")
      return bg ? "Непознат формат на данните — инсталирай индикатора отново" : "Unknown data format — reinstall the indicator";
   if(id == "no_current")
      return bg ? "Няма текуща прогноза (последна: %s)" : "No current forecast (last: %s)";
   if(id == "none")
      return bg ? "Още няма прогнози" : "No forecasts yet";
   if(id == "stale")
      return bg ? "⚠ Данните са стари: %s" : "⚠ Data is old: %s";
   if(id == "shift")
      return bg ? "Часовник на чарта: %s ч спрямо логера" : "Chart clock shift: %s h vs the logger";
   return id;
  }

//+------------------------------------------------------------------+
bool IsGoldSymbol()
  {
   string s = _Symbol;
   StringToUpper(s);
   return StringFind(s, "XAUUSD") == 0 || StringFind(s, "GOLD") == 0;
  }

bool ChartIsUsable()
  {
   return IsGoldSymbol() && PeriodSeconds(PERIOD_CURRENT) <= 3600;
  }

int Col(const string &head[], const int n, const string name)
  {
   for(int i = 0; i < n; i++)
      if(head[i] == name)
         return i;
   return -1;
  }

//+------------------------------------------------------------------+
//| Read the file when it changed. Returns true when marks changed.  |
//| A missing or unreadable file keeps what is already drawn.        |
//+------------------------------------------------------------------+
bool ReadFile()
  {
   if(!FileIsExist(PGS_FILE, FILE_COMMON))
     {
      g_state = 1;
      return false;
     }
   datetime mod = (datetime)FileGetInteger(PGS_FILE, FILE_MODIFY_DATE, true);
   if(g_loaded && g_state == 0 && mod == g_file_mod)
      return false;
   int h = FileOpen(PGS_FILE, FILE_READ | FILE_BIN | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON);
   if(h == INVALID_HANDLE)
     {
      g_state = 1;
      return false;
     }
   uchar bytes[];
   uint got = FileReadArray(h, bytes);
   FileClose(h);
   string text = CharArrayToString(bytes, 0, (int)got, CP_UTF8);

   string lines[];
   int nl = StringSplit(text, '\n', lines);
   string version = "";
   if(nl >= 2 && StringFind(lines[0], "#meta") == 0)
     {
      string meta[];
      int nm = StringSplit(lines[0], '\t', meta);
      for(int i = 1; i < nm; i++)
        {
         int eq = StringFind(meta[i], "=");
         if(eq < 0)
            continue;
         string k = StringSubstr(meta[i], 0, eq);
         string v = StringSubstr(meta[i], eq + 1);
         if(k == "version")
            version = v;
         else if(k == "generated_utc")
            g_generated = StringToTime(v);
         else if(k == "label_utc_offset_min")
            g_label_offset_min = (int)StringToInteger(v);
         else if(k == "written_" + g_lang)
            g_written = v;
         else if(k == "count_" + g_lang)
            g_count = v;
         else if(k == "research_" + g_lang)
            g_research = v;
        }
     }
   string head[];
   int nh = (nl >= 2) ? StringSplit(lines[1], '\t', head) : 0;
   int c_kind = Col(head, nh, "kind"), c_key = Col(head, nh, "key"), c_hour = Col(head, nh, "hour_label");
   int c_end = Col(head, nh, "hour_end_label"), c_dir = Col(head, nh, "direction"), c_out = Col(head, nh, "outcome");
   int c_close = Col(head, nh, "close"), c_text = Col(head, nh, "text_" + g_lang), c_tip = Col(head, nh, "tip_" + g_lang);
   int c_panel = Col(head, nh, "panel_" + g_lang), c_sofia = Col(head, nh, "hour_sofia_" + g_lang);
   if(version != PGS_VERSION || c_kind < 0 || c_key < 0 || c_hour < 0 || c_end < 0 || c_dir < 0 || c_out < 0
      || c_close < 0 || c_text < 0 || c_tip < 0 || c_panel < 0 || c_sofia < 0)
     {
      g_state = 2;
      ArrayResize(g_marks, 0);
      g_file_mod = mod;
      return true;
     }
   ArrayResize(g_marks, 0);
   for(int i = 2; i < nl; i++)
     {
      if(StringLen(lines[i]) == 0)
         continue;
      string f[];
      if(StringSplit(lines[i], '\t', f) < nh)
         continue;
      PgsMark m;
      m.kind = f[c_kind];
      m.key = f[c_key];
      m.hour = StringToTime(f[c_hour]);
      m.hour_end = StringToTime(f[c_end]);
      m.direction = f[c_dir];
      m.outcome = f[c_out];
      m.close = StringToDouble(f[c_close]);
      m.text = f[c_text];
      m.tip = f[c_tip];
      StringReplace(m.tip, "\\n", "\n");
      m.panel = f[c_panel];
      m.hour_sofia = f[c_sofia];
      int k = ArraySize(g_marks);
      ArrayResize(g_marks, k + 1);
      g_marks[k] = m;
     }
   g_state = 0;
   g_loaded = true;
   g_file_mod = mod;
   return true;
  }

//+------------------------------------------------------------------+
//| Seconds to add to a label-clock time to reach the chart's clock. |
//+------------------------------------------------------------------+
int ShiftSeconds()
  {
   long chart_off = (long)(TimeTradeServer() - TimeGMT());
   chart_off = (long)MathRound(chart_off / 900.0) * 900;
   return (int)(chart_off - (long)g_label_offset_min * 60);
  }

double PricePerPixel()
  {
   double top = ChartGetDouble(0, CHART_PRICE_MAX);
   double bottom = ChartGetDouble(0, CHART_PRICE_MIN);
   long px = ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
   if(px <= 0 || top <= bottom)
      return 0;
   return (top - bottom) / (double)px;
  }

void HourRange(const datetime t, const double fallback, double &lo, double &hi)
  {
   lo = fallback;
   hi = fallback;
   int b = iBarShift(_Symbol, PERIOD_H1, t, true);
   if(b < 0)
      return;
   double l = iLow(_Symbol, PERIOD_H1, b), h = iHigh(_Symbol, PERIOD_H1, b);
   if(l > 0 && h >= l)
     {
      lo = l;
      hi = h;
     }
  }

string ForecastDirection(const string key)
  {
   for(int i = ArraySize(g_marks) - 1; i >= 0; i--)
      if(g_marks[i].kind == "forecast" && g_marks[i].key == key)
         return g_marks[i].direction;
   return "";
  }

color OutcomeColor(const string outcome, const color base)
  {
   if(outcome == "right")
      return clrSeaGreen;
   if(outcome == "wrong")
      return clrCrimson;
   if(outcome == "void")
      return clrGray;
   return base;
  }

void Arrow(const string name, const datetime t, const double price, const int code, const color clr,
           const ENUM_ARROW_ANCHOR anchor, const string tip)
  {
   ObjectCreate(0, name, OBJ_ARROW, 0, t, price);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE, code);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, anchor);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
  }

// Vertical text, reading upwards: away from the candle on either side, so the
// texts of neighbouring hours never overlap.
void Text(const string name, const datetime t, const double price, const string text, const color clr,
          const bool below, const string tip)
  {
   ObjectCreate(0, name, OBJ_TEXT, 0, t, price);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, PGS_FONT);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpFontSize);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetDouble(0, name, OBJPROP_ANGLE, 90.0);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, below ? ANCHOR_RIGHT : ANCHOR_LEFT);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
  }

//+------------------------------------------------------------------+
//| Arrows under (up) or over (down) the hour's candle; an alert on  |
//| the side its hour's arrow leaves free.                           |
//+------------------------------------------------------------------+
void DrawMarks()
  {
   ObjectsDeleteAll(0, PGS_PREFIX + "M");
   if(!ChartIsUsable() || g_state == 2)
      return;
   int shift = ShiftSeconds();
   double ppp = PricePerPixel();
   int n = ArraySize(g_marks);
   for(int i = 0; i < n; i++)
     {
      datetime t = g_marks[i].hour + shift;
      double lo, hi;
      HourRange(t, g_marks[i].close, lo, hi);
      double gap = MathMax((hi - lo) * 0.15, 10 * _Point);
      string id = PGS_PREFIX + "M" + IntegerToString(i);
      bool below;
      color clr;
      int code;
      if(g_marks[i].kind == "forecast")
        {
         if(g_marks[i].direction != "up" && g_marks[i].direction != "down")
            continue;
         below = (g_marks[i].direction == "up");
         clr = below ? InpUpColor : InpDownColor;
         code = below ? PGS_CODE_UP : PGS_CODE_DOWN;
        }
      else
        {
         below = (ForecastDirection(g_marks[i].key) == "down");
         clr = InpAlertColor;
         code = PGS_CODE_ALERT;
        }
      double p = below ? lo - gap : hi + gap;
      Arrow(id + "a", t, p, code, clr, below ? ANCHOR_TOP : ANCHOR_BOTTOM, g_marks[i].tip);
      double tp = below ? p - PGS_STEP_PX * ppp : p + PGS_STEP_PX * ppp;
      Text(id + "t", t, tp, g_marks[i].text, OutcomeColor(g_marks[i].outcome, clr), below, g_marks[i].tip);
     }
  }

bool IsStale()
  {
   if(g_generated == 0 || TimeGMT() - g_generated <= PGS_STALE_S)
      return false;
   datetime last_tick = (datetime)SymbolInfoInteger(_Symbol, SYMBOL_TIME);
   return TimeTradeServer() - last_tick < PGS_TRADING_S;
  }

string ShiftText(const int shift)
  {
   string s = DoubleToString(MathAbs(shift) / 3600.0, (shift % 3600 == 0) ? 0 : 1);
   return StringFormat(T("shift"), (shift > 0 ? "+" : "−") + s);
  }

void AddLine(string &lines[], color &colors[], const string text, const color clr)
  {
   if(StringLen(text) == 0)
      return;
   int k = ArraySize(lines);
   ArrayResize(lines, k + 1);
   ArrayResize(colors, k + 1);
   lines[k] = text;
   colors[k] = clr;
  }

//+------------------------------------------------------------------+
void DrawPanel()
  {
   ObjectsDeleteAll(0, PGS_PREFIX + "P");
   color fg = (color)ChartGetInteger(0, CHART_COLOR_FOREGROUND);
   string lines[];
   color colors[];
   AddLine(lines, colors, T("title"), fg);
   if(!IsGoldSymbol())
      AddLine(lines, colors, T("not_gold"), fg);
   else if(PeriodSeconds(PERIOD_CURRENT) > 3600)
      AddLine(lines, colors, T("timeframe"), fg);
   else if(g_state == 2)
      AddLine(lines, colors, T("unknown"), clrOrangeRed);
   else
     {
      if(g_state == 1)
         AddLine(lines, colors, T("missing"), clrOrangeRed);
      if(g_loaded)
        {
         int shift = ShiftSeconds();
         datetime now = TimeTradeServer();
         int cur = -1, last = -1, alert = -1;
         for(int i = 0; i < ArraySize(g_marks); i++)
           {
            bool open = (g_marks[i].hour_end + shift > now);
            if(g_marks[i].kind == "forecast")
              {
               if(last < 0 || g_marks[i].hour > g_marks[last].hour)
                  last = i;
               if(open && (cur < 0 || g_marks[i].hour > g_marks[cur].hour))
                  cur = i;
              }
            else if(open && (alert < 0 || g_marks[i].hour > g_marks[alert].hour))
               alert = i;
           }
         if(cur >= 0)
            AddLine(lines, colors, g_marks[cur].panel, g_marks[cur].direction == "up" ? InpUpColor : InpDownColor);
         else if(last >= 0)
            AddLine(lines, colors, StringFormat(T("no_current"), g_marks[last].hour_sofia), fg);
         else
            AddLine(lines, colors, T("none"), fg);
         if(alert >= 0)
            AddLine(lines, colors, g_marks[alert].panel, InpAlertColor);
         AddLine(lines, colors, g_count, fg);
         if(IsStale())
            AddLine(lines, colors, StringFormat(T("stale"), g_written), clrOrangeRed);
         else
            AddLine(lines, colors, g_written, fg);
         if(shift != 0)
            AddLine(lines, colors, ShiftText(shift), fg);
         AddLine(lines, colors, g_research, fg);
        }
     }
   bool right = (InpCorner == CORNER_RIGHT_UPPER || InpCorner == CORNER_RIGHT_LOWER);
   bool lower = (InpCorner == CORNER_LEFT_LOWER || InpCorner == CORNER_RIGHT_LOWER);
   ENUM_ANCHOR_POINT anchor = lower ? (right ? ANCHOR_RIGHT_LOWER : ANCHOR_LEFT_LOWER)
                                    : (right ? ANCHOR_RIGHT_UPPER : ANCHOR_LEFT_UPPER);
   int n = ArraySize(lines);
   int step = InpFontSize * 2 + 2;
   for(int i = 0; i < n; i++)
     {
      string name = PGS_PREFIX + "P" + IntegerToString(i);
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, InpCorner);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, anchor);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 10);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 20 + (lower ? n - 1 - i : i) * step);
      ObjectSetString(0, name, OBJPROP_TEXT, lines[i]);
      ObjectSetString(0, name, OBJPROP_FONT, PGS_FONT);
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpFontSize);
      ObjectSetInteger(0, name, OBJPROP_COLOR, colors[i]);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
     }
  }

void Redraw()
  {
   DrawMarks();
   DrawPanel();
   ChartRedraw();
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   g_lang = (InpLanguage == PGS_BG) ? "bg" : "en";
   IndicatorSetString(INDICATOR_SHORTNAME, "ProphetGoldSignals");
   ReadFile();
   Redraw();
   EventSetTimer(30);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, PGS_PREFIX);
   ChartRedraw();
  }

// Every 30 s: pick up a rewritten file, and redraw anyway -- the current
// forecast, an open alert, staleness and the live hour's candle all move with
// the clock, not with the file.
void OnTimer()
  {
   ReadFile();
   Redraw();
  }

int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
  {
   datetime bar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(bar != g_last_bar)
     {
      g_last_bar = bar;
      Redraw();
     }
   return rates_total;
  }

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(id == CHARTEVENT_CHART_CHANGE)
      Redraw();
  }
//+------------------------------------------------------------------+
