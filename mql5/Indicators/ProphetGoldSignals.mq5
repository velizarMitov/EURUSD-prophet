//+------------------------------------------------------------------+
//| ProphetGoldSignals.mq5                                           |
//| EUR/USD Prophet gold signals on the chart (openspec change       |
//| gold-mt5-chart-signals, spec gold-forecasting/mt5-chart-signals). |
//|                                                                  |
//| Reads Common\Files\EURUSDProphet\gold_signals.tsv, which the     |
//| gold logger rebuilds every hour (src/gold/mt5_export.py), and    |
//| draws on a GOLD / XAUUSD chart:                                  |
//|   H1 and lower - an up/down arrow on each hour h1_lstm forecast, |
//|                  a two-headed marker on each alerted hour;       |
//|   D1           - a d1_lstm (solid) and a d1_gbm (hollow) arrow   |
//|                  on each forecast day;                           |
//| each with a short text and a hover description, plus a corner    |
//| panel and buttons that hide arrows below a chosen probability.   |
//| Every word comes from the file except the status lines in T().   |
//| An indicator: it reads one file, draws, and remembers the        |
//| chart's filter choice, nothing else.                             |
//+------------------------------------------------------------------+
#property copyright   "EUR/USD Prophet"
#property version     "2.10"
#property description "Gold research signals: h1_lstm, d1_lstm and d1_gbm direction, large-move alerts."
#property description "Research - no proven edge, gross of cost."
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

enum ENUM_PGS_LANG
  {
   PGS_BG = 0, // Български
   PGS_EN = 1  // English
  };

input ENUM_PGS_LANG    InpLanguage       = PGS_BG;             // Език / Language
input int              InpMinProbability = 50;                 // Arrows from probability, % (50 = all)
input color            InpUpColor        = clrDodgerBlue;      // Up arrow colour (h1_lstm, d1_lstm)
input color            InpDownColor      = clrOrangeRed;       // Down arrow colour (h1_lstm, d1_lstm)
input color            InpGbmColor       = clrMediumOrchid;    // d1_gbm arrow colour
input color            InpAlertColor     = clrGold;            // Large-move marker colour
input ENUM_BASE_CORNER InpCorner         = CORNER_RIGHT_UPPER; // Panel corner
input int              InpFontSize       = 8;                  // Font size

#define PGS_PREFIX     "PGS_"
#define PGS_FILE       "EURUSDProphet\\gold_signals.tsv"
#define PGS_VERSION    "2"
#define PGS_CODE_UP    233
#define PGS_CODE_DOWN  234
#define PGS_CODE_UP2   241
#define PGS_CODE_DOWN2 242
#define PGS_CODE_ALERT 244
#define PGS_STALE_S    7200
#define PGS_TRADING_S  600
#define PGS_STEP_PX    18
#define PGS_BTN_W      46
#define PGS_BTN_H      18
#define PGS_FONT       "Segoe UI"

#define PGS_MODE_NONE  0
#define PGS_MODE_H1    1
#define PGS_MODE_D1    2

int    g_filters[4] = {50, 52, 54, 56};
string g_h1_cells[1] = {"h1_lstm"};
string g_d1_cells[2] = {"d1_lstm", "d1_gbm"};

struct PgsMark
  {
   string            kind;
   string            key;
   string            cell;
   string            cadence;
   datetime          hour;       // bar-label clock (the logger's Europe/Berlin)
   datetime          hour_end;
   string            direction;
   int               prob;
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
string   g_written = "", g_count_h1 = "", g_count_d1 = "", g_filter = "", g_research = "";
string   g_second = "", g_second_tip = "";   // external models' line, empty when none is current
string   g_lang = "bg";
datetime g_last_bar = 0;
int      g_min = 50;

//+------------------------------------------------------------------+
//| The status lines the file cannot carry, BG and EN.               |
//+------------------------------------------------------------------+
string T(const string id)
  {
   bool bg = (InpLanguage == PGS_BG);
   if(id == "title")
      return bg ? "Злато · h1_lstm, d1_lstm, d1_gbm" : "Gold · h1_lstm, d1_lstm, d1_gbm";
   if(id == "not_gold")
      return bg ? "Това не е графика на злато — няма сигнали" : "This chart is not gold — no signals";
   if(id == "timeframe")
      return bg ? "Сигналите се показват на H1 или по-малък таймфрейм и на D1" : "Signals show on H1 or lower, and on D1";
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

int ChartMode()
  {
   if(!IsGoldSymbol())
      return PGS_MODE_NONE;
   if(PeriodSeconds(PERIOD_CURRENT) <= 3600)
      return PGS_MODE_H1;
   if(_Period == PERIOD_D1)
      return PGS_MODE_D1;
   return PGS_MODE_NONE;
  }

int Col(const string &head[], const int n, const string name)
  {
   for(int i = 0; i < n; i++)
      if(head[i] == name)
         return i;
   return -1;
  }

//+------------------------------------------------------------------+
//| The filter, kept per chart across timeframe changes (which       |
//| reinitialise the indicator). A changed input setting wins.       |
//+------------------------------------------------------------------+
string GvName(const string what)
  {
   return PGS_PREFIX + what + "_" + IntegerToString(ChartID());
  }

void LoadFilter()
  {
   string inp = GvName("inp"), cur = GvName("min");
   g_min = InpMinProbability;
   if(GlobalVariableCheck(cur) && GlobalVariableCheck(inp) && (int)GlobalVariableGet(inp) == InpMinProbability)
      g_min = (int)GlobalVariableGet(cur);
   GlobalVariableSet(inp, InpMinProbability);
   GlobalVariableSet(cur, g_min);
  }

void SetFilter(const int value)
  {
   g_min = value;
   GlobalVariableSet(GvName("min"), g_min);
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
         else if(k == "count_h1_" + g_lang)
            g_count_h1 = v;
         else if(k == "count_d1_" + g_lang)
            g_count_d1 = v;
         else if(k == "filter_" + g_lang)
            g_filter = v;
         else if(k == "research_" + g_lang)
            g_research = v;
         else if(k == "second_" + g_lang)
            g_second = v;
         else if(k == "second_tip_" + g_lang)
            g_second_tip = v;
        }
     }
   string head[];
   int nh = (nl >= 2) ? StringSplit(lines[1], '\t', head) : 0;
   int c_kind = Col(head, nh, "kind"), c_key = Col(head, nh, "key"), c_cell = Col(head, nh, "cell");
   int c_cad = Col(head, nh, "cadence"), c_hour = Col(head, nh, "hour_label"), c_end = Col(head, nh, "hour_end_label");
   int c_dir = Col(head, nh, "direction"), c_prob = Col(head, nh, "prob_pct"), c_out = Col(head, nh, "outcome");
   int c_close = Col(head, nh, "close"), c_text = Col(head, nh, "text_" + g_lang), c_tip = Col(head, nh, "tip_" + g_lang);
   int c_panel = Col(head, nh, "panel_" + g_lang), c_sofia = Col(head, nh, "hour_sofia_" + g_lang);
   if(version != PGS_VERSION || c_kind < 0 || c_key < 0 || c_cell < 0 || c_cad < 0 || c_hour < 0 || c_end < 0
      || c_dir < 0 || c_prob < 0 || c_out < 0 || c_close < 0 || c_text < 0 || c_tip < 0 || c_panel < 0 || c_sofia < 0)
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
      m.cell = f[c_cell];
      m.cadence = f[c_cad];
      m.hour = StringToTime(f[c_hour]);
      m.hour_end = StringToTime(f[c_end]);
      m.direction = f[c_dir];
      m.prob = (int)StringToInteger(f[c_prob]);
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

void BarRange(const ENUM_TIMEFRAMES tf, const int b, const double fallback, double &lo, double &hi)
  {
   lo = fallback;
   hi = fallback;
   if(b < 0)
      return;
   double l = iLow(_Symbol, tf, b), h = iHigh(_Symbol, tf, b);
   if(l > 0 && h >= l)
     {
      lo = l;
      hi = h;
     }
  }

// A direction arrow the chart shows: its cadence, at or above the filter.
bool Shown(const int i, const int mode)
  {
   if(g_marks[i].kind != "forecast" || (g_marks[i].direction != "up" && g_marks[i].direction != "down"))
      return false;
   string cad = (mode == PGS_MODE_D1) ? "D1" : "H1";
   return mode != PGS_MODE_NONE && g_marks[i].cadence == cad && g_marks[i].prob >= g_min;
  }

string ShownDirection(const string key, const int mode)
  {
   for(int i = ArraySize(g_marks) - 1; i >= 0; i--)
      if(g_marks[i].key == key && Shown(i, mode))
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

color ArrowColor(const int i)
  {
   if(g_marks[i].cell == "d1_gbm")
      return InpGbmColor;
   return g_marks[i].direction == "up" ? InpUpColor : InpDownColor;
  }

int ArrowCode(const int i)
  {
   bool up = (g_marks[i].direction == "up");
   if(g_marks[i].cell == "d1_gbm")
      return up ? PGS_CODE_UP2 : PGS_CODE_DOWN2;
   return up ? PGS_CODE_UP : PGS_CODE_DOWN;
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
// texts of neighbouring bars never overlap.
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
//| H1 and lower: an arrow under (up) or over (down) the hour's      |
//| candle; an alert on the side its hour's arrow leaves free.       |
//+------------------------------------------------------------------+
void DrawHourly(const double ppp)
  {
   int shift = ShiftSeconds();
   for(int i = 0; i < ArraySize(g_marks); i++)
     {
      bool alert = (g_marks[i].kind == "alert" && g_marks[i].cadence == "H1");
      if(!alert && !Shown(i, PGS_MODE_H1))
         continue;
      datetime t = g_marks[i].hour + shift;
      double lo, hi;
      BarRange(PERIOD_H1, iBarShift(_Symbol, PERIOD_H1, t, true), g_marks[i].close, lo, hi);
      double gap = MathMax((hi - lo) * 0.15, 10 * _Point);
      bool below = alert ? (ShownDirection(g_marks[i].key, PGS_MODE_H1) == "down") : (g_marks[i].direction == "up");
      color clr = alert ? InpAlertColor : ArrowColor(i);
      int code = alert ? PGS_CODE_ALERT : ArrowCode(i);
      string id = PGS_PREFIX + "M" + IntegerToString(i);
      double p = below ? lo - gap : hi + gap;
      Arrow(id + "a", t, p, code, clr, below ? ANCHOR_TOP : ANCHOR_BOTTOM, g_marks[i].tip);
      double tp = below ? p - PGS_STEP_PX * ppp : p + PGS_STEP_PX * ppp;
      Text(id + "t", t, tp, g_marks[i].text, OutcomeColor(g_marks[i].outcome, clr), below, g_marks[i].tip);
     }
  }

//+------------------------------------------------------------------+
//| D1: per day and side, the shown cells' arrows stack next to the  |
//| candle and one text joins their short texts.                     |
//+------------------------------------------------------------------+
void DrawDaily(const double ppp)
  {
   int n = ArraySize(g_marks);
   int done[];
   ArrayResize(done, n);
   ArrayInitialize(done, 0);
   for(int i = 0; i < n; i++)
     {
      if(done[i] != 0 || !Shown(i, PGS_MODE_D1))
         continue;
      bool below = (g_marks[i].direction == "up");
      int group[];
      for(int c = 0; c < ArraySize(g_d1_cells); c++)
         for(int j = i; j < n; j++)
            if(done[j] == 0 && Shown(j, PGS_MODE_D1) && g_marks[j].hour == g_marks[i].hour
               && (g_marks[j].direction == "up") == below && g_marks[j].cell == g_d1_cells[c])
              {
               done[j] = 1;
               int k = ArraySize(group);
               ArrayResize(group, k + 1);
               group[k] = j;
              }
      if(ArraySize(group) == 0)
         continue;
      // The chart's own candle for that calendar day; a clock a few hours off Berlin stays on the day.
      int b = iBarShift(_Symbol, PERIOD_D1, g_marks[i].hour + 12 * 3600, false);
      datetime t = g_marks[i].hour;
      if(b >= 0 && MathAbs((double)(iTime(_Symbol, PERIOD_D1, b) - g_marks[i].hour)) <= 6 * 3600)
         t = iTime(_Symbol, PERIOD_D1, b);
      else
         b = -1;
      double lo, hi;
      BarRange(PERIOD_D1, b, g_marks[group[0]].close, lo, hi);
      double gap = MathMax((hi - lo) * 0.05, 10 * _Point);
      double p = below ? lo - gap : hi + gap;
      string text = "", tip = "";
      for(int k = 0; k < ArraySize(group); k++)
        {
         int j = group[k];
         double pk = below ? p - k * PGS_STEP_PX * ppp : p + k * PGS_STEP_PX * ppp;
         Arrow(PGS_PREFIX + "M" + IntegerToString(j) + "a", t, pk, ArrowCode(j), ArrowColor(j),
               below ? ANCHOR_TOP : ANCHOR_BOTTOM, g_marks[j].tip);
         text += (k > 0 ? "  |  " : "") + g_marks[j].text;
         tip += (k > 0 ? "\n\n" : "") + g_marks[j].tip;
        }
      int last = ArraySize(group);
      double tp = below ? p - last * PGS_STEP_PX * ppp : p + last * PGS_STEP_PX * ppp;
      color fg = (color)ChartGetInteger(0, CHART_COLOR_FOREGROUND);
      color clr = (last == 1) ? OutcomeColor(g_marks[group[0]].outcome, ArrowColor(group[0])) : fg;
      Text(PGS_PREFIX + "M" + IntegerToString(group[0]) + "t", t, tp, text, clr, below, tip);
     }
  }

void DrawMarks()
  {
   ObjectsDeleteAll(0, PGS_PREFIX + "M");
   if(g_state == 2)
      return;
   int mode = ChartMode();
   double ppp = PricePerPixel();
   if(mode == PGS_MODE_H1)
      DrawHourly(ppp);
   else if(mode == PGS_MODE_D1)
      DrawDaily(ppp);
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

// The Python count template, filled for the arrows this chart shows.
string CountLine(const int mode)
  {
   string s = (mode == PGS_MODE_D1) ? g_count_d1 : g_count_h1;
   StringReplace(s, "{min}", IntegerToString(g_min));
   int nc = (mode == PGS_MODE_D1) ? ArraySize(g_d1_cells) : ArraySize(g_h1_cells);
   for(int c = 0; c < nc; c++)
     {
      string cell = (mode == PGS_MODE_D1) ? g_d1_cells[c] : g_h1_cells[c];
      int right = 0, settled = 0;
      for(int i = 0; i < ArraySize(g_marks); i++)
         if(g_marks[i].cell == cell && Shown(i, mode) && (g_marks[i].outcome == "right" || g_marks[i].outcome == "wrong"))
           {
            settled++;
            if(g_marks[i].outcome == "right")
               right++;
           }
      StringReplace(s, "{r_" + cell + "}", IntegerToString(right));
      StringReplace(s, "{t_" + cell + "}", IntegerToString(settled));
     }
   return s;
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

// Current-forecast lines: the newest open forecast per cell of the chart's cadence, unfiltered.
void CurrentLines(string &lines[], color &colors[], const int mode, const color fg)
  {
   int shift = ShiftSeconds();
   datetime now = TimeTradeServer();
   int nc = (mode == PGS_MODE_D1) ? ArraySize(g_d1_cells) : ArraySize(g_h1_cells);
   string cad = (mode == PGS_MODE_D1) ? "D1" : "H1";
   int last = -1, shown = 0;
   for(int c = 0; c < nc; c++)
     {
      string cell = (mode == PGS_MODE_D1) ? g_d1_cells[c] : g_h1_cells[c];
      int cur = -1;
      for(int i = 0; i < ArraySize(g_marks); i++)
        {
         if(g_marks[i].kind != "forecast" || g_marks[i].cadence != cad || g_marks[i].cell != cell)
            continue;
         if(last < 0 || g_marks[i].hour > g_marks[last].hour)
            last = i;
         if(g_marks[i].hour_end + shift > now && (cur < 0 || g_marks[i].hour > g_marks[cur].hour))
            cur = i;
        }
      if(cur >= 0)
        {
         AddLine(lines, colors, g_marks[cur].panel, ArrowColor(cur));
         shown++;
        }
     }
   if(shown == 0 && last >= 0)
      AddLine(lines, colors, StringFormat(T("no_current"), g_marks[last].hour_sofia), fg);
   else if(shown == 0)
      AddLine(lines, colors, T("none"), fg);
  }

void DrawButtons(const bool right, const bool lower)
  {
   ObjectsDeleteAll(0, PGS_PREFIX + "B");
   for(int i = 0; i < ArraySize(g_filters); i++)
     {
      string name = PGS_PREFIX + "B" + IntegerToString(g_filters[i]);
      int x = right ? 10 + PGS_BTN_W + (ArraySize(g_filters) - 1 - i) * (PGS_BTN_W + 4) : 10 + i * (PGS_BTN_W + 4);
      ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, InpCorner);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, lower ? 20 + PGS_BTN_H : 20);
      ObjectSetInteger(0, name, OBJPROP_XSIZE, PGS_BTN_W);
      ObjectSetInteger(0, name, OBJPROP_YSIZE, PGS_BTN_H);
      ObjectSetString(0, name, OBJPROP_TEXT, "≥" + IntegerToString(g_filters[i]) + " %");
      ObjectSetString(0, name, OBJPROP_FONT, PGS_FONT);
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpFontSize);
      ObjectSetInteger(0, name, OBJPROP_STATE, g_filters[i] == g_min);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
     }
  }

//+------------------------------------------------------------------+
void DrawPanel()
  {
   ObjectsDeleteAll(0, PGS_PREFIX + "P");
   ObjectsDeleteAll(0, PGS_PREFIX + "B");
   color fg = (color)ChartGetInteger(0, CHART_COLOR_FOREGROUND);
   int mode = ChartMode();
   string lines[];
   color colors[];
   int second_at = -1;
   AddLine(lines, colors, T("title"), fg);
   if(!IsGoldSymbol())
      AddLine(lines, colors, T("not_gold"), fg);
   else if(mode == PGS_MODE_NONE)
      AddLine(lines, colors, T("timeframe"), fg);
   else if(g_state == 2)
      AddLine(lines, colors, T("unknown"), clrOrangeRed);
   else
     {
      if(g_state == 1)
         AddLine(lines, colors, T("missing"), clrOrangeRed);
      if(g_loaded)
        {
         CurrentLines(lines, colors, mode, fg);
         if(mode == PGS_MODE_H1)
           {
            int shift = ShiftSeconds(), alert = -1;
            datetime now = TimeTradeServer();
            for(int i = 0; i < ArraySize(g_marks); i++)
               if(g_marks[i].kind == "alert" && g_marks[i].hour_end + shift > now
                  && (alert < 0 || g_marks[i].hour > g_marks[alert].hour))
                  alert = i;
            if(alert >= 0)
               AddLine(lines, colors, g_marks[alert].panel, InpAlertColor);
            if(StringLen(g_second) > 0)
              {
               second_at = ArraySize(lines);
               AddLine(lines, colors, g_second, fg);
              }
           }
         if(g_min > 50)
           {
            string f = g_filter;
            StringReplace(f, "{min}", IntegerToString(g_min));
            AddLine(lines, colors, f, fg);
           }
         AddLine(lines, colors, CountLine(mode), fg);
         if(IsStale())
            AddLine(lines, colors, StringFormat(T("stale"), g_written), clrOrangeRed);
         else
            AddLine(lines, colors, g_written, fg);
         int shift = ShiftSeconds();
         if(mode == PGS_MODE_H1 && shift != 0)
            AddLine(lines, colors, ShiftText(shift), fg);
         AddLine(lines, colors, g_research, fg);
        }
     }
   bool right = (InpCorner == CORNER_RIGHT_UPPER || InpCorner == CORNER_RIGHT_LOWER);
   bool lower = (InpCorner == CORNER_LEFT_LOWER || InpCorner == CORNER_RIGHT_LOWER);
   bool buttons = (mode != PGS_MODE_NONE);
   if(buttons)
      DrawButtons(right, lower);
   ENUM_ANCHOR_POINT anchor = lower ? (right ? ANCHOR_RIGHT_LOWER : ANCHOR_LEFT_LOWER)
                                    : (right ? ANCHOR_RIGHT_UPPER : ANCHOR_LEFT_UPPER);
   int n = ArraySize(lines);
   int step = InpFontSize * 2 + 2;
   int top = 20 + (buttons ? PGS_BTN_H + 6 : 0);
   for(int i = 0; i < n; i++)
     {
      string name = PGS_PREFIX + "P" + IntegerToString(i);
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, InpCorner);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, anchor);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 10);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, top + (lower ? n - 1 - i : i) * step);
      ObjectSetString(0, name, OBJPROP_TEXT, lines[i]);
      if(i == second_at)
         ObjectSetString(0, name, OBJPROP_TOOLTIP, g_second_tip);
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
   LoadFilter();
   ReadFile();
   Redraw();
   EventSetTimer(30);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, PGS_PREFIX);
   if(reason == REASON_REMOVE)
     {
      GlobalVariableDel(GvName("min"));
      GlobalVariableDel(GvName("inp"));
     }
   ChartRedraw();
  }

// Every 30 s: pick up a rewritten file, and redraw anyway -- the current
// forecast, an open alert, staleness and the live candle all move with the
// clock, not with the file.
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
   else if(id == CHARTEVENT_OBJECT_CLICK && StringFind(sparam, PGS_PREFIX + "B") == 0)
     {
      SetFilter((int)StringToInteger(StringSubstr(sparam, StringLen(PGS_PREFIX) + 1)));
      Redraw();
     }
  }
//+------------------------------------------------------------------+
