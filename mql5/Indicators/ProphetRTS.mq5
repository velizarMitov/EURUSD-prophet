//+------------------------------------------------------------------+
//| ProphetRTS.mq5                                                    |
//| EUR/USD Prophet - Rockefeller Treasury Services spot levels       |
//| (openspec change rts-signals, spec rts-signals/chart-display).    |
//|                                                                   |
//| DISPLAY ONLY. Reads Common\Files\EURUSDProphet\rts_signals.tsv,   |
//| which python -m src.rts_signals.run rebuilds after every pass.    |
//| Python computes every price, time and text; this file only draws. |
//| It never places, modifies or closes an order.                     |
//|                                                                   |
//| The levels are RTS's suggestions, not this project's signal.      |
//| today   entry / stop / target lines and a direction arrow,        |
//|         labelled unverified (owner decision 2026-11-28)           |
//| trade   past base trades: entry-to-exit line coloured by the net  |
//|         result, stop and target up to the exit, or the reason     |
//|         there was no trade                                        |
//| morning RTS's Morning report positions in the corner              |
//| Drawn on EURUSD, GBPUSD, AUDUSD, USDJPY, USDCHF, USDCAD, M1-H1.   |
//+------------------------------------------------------------------+
#property copyright   "EUR/USD Prophet"
#property version     "1.00"
#property description "RTS spot levels and their outcome on your prices (display only)."
#property description "Third-party suggestions; no verdict until the forward test is scored."
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

#define RTS_FILE   "EURUSDProphet\\rts_signals.tsv"
#define RTS_PREFIX "PRTS_"

enum ENUM_RTS_LANG { RTS_BG = 0, RTS_EN = 1 };

input ENUM_RTS_LANG    InpLanguage    = RTS_BG;            // Език / Language
input color            InpEntryColor  = clrDodgerBlue;     // Entry level
input color            InpStopColor   = clrTomato;         // Stop level
input color            InpTargetColor = clrMediumSeaGreen; // Target level
input color            InpWinColor    = clrLimeGreen;      // Past trade, net gain
input color            InpLossColor   = clrOrangeRed;      // Past trade, net loss
input color            InpTextColor   = clrSilver;         // Corner text
input bool             InpShowPast    = true;              // Show past trades
input ENUM_BASE_CORNER InpCorner      = CORNER_LEFT_UPPER; // Status corner
input int              InpFontSize    = 8;                 // Font size

datetime g_file_mod = 0;
string   g_pairs[] = {"EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF", "USDCAD"};

//+------------------------------------------------------------------+
string Pick(const string bg, const string en)
  {
   return (InpLanguage == RTS_BG) ? bg : en;
  }

bool SymbolIs(const string sym)
  {
   return sym != "" && StringFind(_Symbol, sym) == 0;
  }

bool SymbolShown()
  {
   for(int i = 0; i < ArraySize(g_pairs); i++)
      if(SymbolIs(g_pairs[i]))
         return true;
   return false;
  }

bool TimeframeShown()
  {
   ENUM_TIMEFRAMES tf = (ENUM_TIMEFRAMES)Period();
   return tf == PERIOD_M1 || tf == PERIOD_M5 || tf == PERIOD_M15 || tf == PERIOD_M30 || tf == PERIOD_H1;
  }

//+------------------------------------------------------------------+
void Line(const string name, datetime t1, double p1, datetime t2, double p2, color c, ENUM_LINE_STYLE st, int w)
  {
   ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_STYLE, st);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, w);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_RAY_LEFT, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

void Arrow(const string name, datetime t, double p, bool up, color c)
  {
   ObjectCreate(0, name, up ? OBJ_ARROW_UP : OBJ_ARROW_DOWN, 0, t, p);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, up ? ANCHOR_TOP : ANCHOR_BOTTOM);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

void Text(const string name, datetime t, double p, const string txt, color c)
  {
   ObjectCreate(0, name, OBJ_TEXT, 0, t, p);
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpFontSize);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

void CornerLabel(const string name, int line, const string txt, color c)
  {
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, InpCorner);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 8);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 20 + line * (InpFontSize + 8));
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpFontSize);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

//+------------------------------------------------------------------+
int Col(string &head[], const string name)
  {
   for(int i = 0; i < ArraySize(head); i++)
      if(head[i] == name)
         return i;
   return -1;
  }

string Get(string &f[], string &head[], const string name)
  {
   int i = Col(head, name);
   if(i < 0 || i >= ArraySize(f))
      return "";
   return f[i];
  }

double Num(string &f[], string &head[], const string name)
  {
   return StringToDouble(Get(f, head, name));
  }

//+------------------------------------------------------------------+
//| Today's RTS levels: entry, stop, target and a direction arrow.   |
//+------------------------------------------------------------------+
void DrawToday(string &f[], string &head[], const string id)
  {
   datetime t0 = StringToTime(Get(f, head, "start"));
   datetime t1 = StringToTime(Get(f, head, "end"));
   double e = Num(f, head, "entry"), s = Num(f, head, "stop"), g = Num(f, head, "target");
   bool buy = Get(f, head, "direction") == "BUY";
   Line(id + "e", t0, e, t1, e, InpEntryColor, STYLE_SOLID, 2);
   Line(id + "s", t0, s, t1, s, InpStopColor, STYLE_DASH, 1);
   Line(id + "g", t0, g, t1, g, InpTargetColor, STYLE_DASH, 1);
   Arrow(id + "a", t0, e, buy, InpEntryColor);
   Text(id + "l", t0, buy ? g : s, Pick(Get(f, head, "label_bg"), Get(f, head, "label_en")), InpEntryColor);
  }

//+------------------------------------------------------------------+
//| A past base trade, or the reason it did not happen.              |
//+------------------------------------------------------------------+
void DrawTrade(string &f[], string &head[], const string id)
  {
   string lab = Pick(Get(f, head, "label_bg"), Get(f, head, "label_en"));
   datetime t0 = StringToTime(Get(f, head, "start"));
   datetime t1 = StringToTime(Get(f, head, "end"));
   double e = Num(f, head, "entry");
   if(Get(f, head, "status") != "filled")
     {
      if(e > 0)
        {
         Line(id + "e", t0, e, t1, e, InpEntryColor, STYLE_DOT, 1);
         Text(id + "l", t0, e, lab, InpTextColor);
        }
      return;
     }
   datetime te = StringToTime(Get(f, head, "entry_time"));
   datetime tx = StringToTime(Get(f, head, "exit_time"));
   double x = Num(f, head, "exit"), s = Num(f, head, "stop"), g = Num(f, head, "target");
   color c = Num(f, head, "net_r") >= 0 ? InpWinColor : InpLossColor;
   Line(id + "s", te, s, tx, s, InpStopColor, STYLE_DOT, 1);
   Line(id + "g", te, g, tx, g, InpTargetColor, STYLE_DOT, 1);
   Line(id + "t", te, e, tx, x, c, STYLE_SOLID, 2);
   Text(id + "l", tx, x, lab, c);
  }

//+------------------------------------------------------------------+
//| Read the file when it changed and redraw everything.             |
//+------------------------------------------------------------------+
void Refresh(bool force)
  {
   if(!SymbolShown())
     {
      ObjectsDeleteAll(0, RTS_PREFIX);
      CornerLabel(RTS_PREFIX + "c0", 0, Pick("Този индикатор е за EURUSD, GBPUSD, AUDUSD, USDJPY, USDCHF, USDCAD.",
                                             "This indicator is for EURUSD, GBPUSD, AUDUSD, USDJPY, USDCHF, USDCAD."),
                  InpTextColor);
      ChartRedraw();
      return;
     }
   if(!FileIsExist(RTS_FILE, FILE_COMMON))
     {
      ObjectsDeleteAll(0, RTS_PREFIX);
      CornerLabel(RTS_PREFIX + "c0", 0, Pick("Няма файл rts_signals.tsv (пуснете RTS задачата).",
                                             "No rts_signals.tsv yet (run the RTS task)."), InpTextColor);
      ChartRedraw();
      return;
     }
   datetime mod = (datetime)FileGetInteger(RTS_FILE, FILE_MODIFY_DATE, true);
   if(!force && mod == g_file_mod)
      return;
   int h = FileOpen(RTS_FILE, FILE_READ | FILE_BIN | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON);
   if(h == INVALID_HANDLE)
      return;                                   // keep what is drawn; retry on the next timer
   uchar bytes[];
   uint got = FileReadArray(h, bytes);
   FileClose(h);
   g_file_mod = mod;
   string text = CharArrayToString(bytes, 0, (int)got, CP_UTF8);
   string lines[];
   int nl = StringSplit(text, '\n', lines);

   ObjectsDeleteAll(0, RTS_PREFIX);
   string status = "", note = "", mtitle = "";
   string head[];
   int nh = 0, corner = 2, k = 0;
   string morning[];
   ArrayResize(morning, 0);
   for(int i = 0; i < nl; i++)
     {
      string ln = lines[i];
      StringReplace(ln, "\r", "");
      if(ln == "")
         continue;
      string f[];
      int nf = StringSplit(ln, '\t', f);
      if(f[0] == "#meta")
         continue;
      if(f[0] == "#text")
        {
         if(nf >= 4 && f[1] == "status")
            status = Pick(f[2], f[3]);
         if(nf >= 4 && f[1] == "note")
            note = Pick(f[2], f[3]);
         if(nf >= 4 && f[1] == "morning")
            mtitle = Pick(f[2], f[3]);
         continue;
        }
      if(nh == 0)
        {
         nh = StringSplit(ln, '\t', head);
         continue;
        }
      string kind = Get(f, head, "kind");
      string lab = Pick(Get(f, head, "label_bg"), Get(f, head, "label_en"));
      if(kind == "morning")
        {
         int m = ArraySize(morning);
         ArrayResize(morning, m + 1);
         morning[m] = lab;
         continue;
        }
      if(!SymbolIs(Get(f, head, "symbol")))
         continue;
      string id = RTS_PREFIX + IntegerToString(k++) + "_";
      if(kind == "today")
        {
         CornerLabel(RTS_PREFIX + "t" + IntegerToString(k), corner++, lab, InpEntryColor);
         if(Get(f, head, "status") != "npr" && TimeframeShown())
            DrawToday(f, head, id);
        }
      else if(kind == "trade" && InpShowPast && TimeframeShown())
         DrawTrade(f, head, id);
     }
   CornerLabel(RTS_PREFIX + "c0", 0, status, InpTextColor);
   CornerLabel(RTS_PREFIX + "c1", 1, note, InpTextColor);
   if(!TimeframeShown())
      CornerLabel(RTS_PREFIX + "tf", corner++, Pick("Рисува се само на M1-H1.", "Drawn on M1-H1 only."), InpTextColor);
   if(ArraySize(morning) > 0)
     {
      CornerLabel(RTS_PREFIX + "mt", corner++, mtitle, InpTextColor);
      for(int j = 0; j < ArraySize(morning); j++)
         CornerLabel(RTS_PREFIX + "m" + IntegerToString(j), corner++, morning[j], InpTextColor);
     }
   ChartRedraw();
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(30);
   Refresh(true);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, RTS_PREFIX);
   ChartRedraw();
  }

void OnTimer()
  {
   Refresh(false);
  }

int OnCalculate(const int rates_total, const int prev_calculated, const datetime &time[], const double &open[],
                const double &high[], const double &low[], const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
  {
   return rates_total;
  }
//+------------------------------------------------------------------+
