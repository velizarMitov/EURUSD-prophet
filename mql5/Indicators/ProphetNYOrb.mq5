//+------------------------------------------------------------------+
//| ProphetNYOrb.mq5                                                  |
//| EUR/USD Prophet - New York opening range, range forecast and the  |
//| breakout rule (openspec change fx-ny-orb-hot, spec                |
//| fx-ny-orb-hot/chart-display).                                     |
//|                                                                   |
//| DISPLAY ONLY. Reads Common\Files\EURUSDProphet\ny_orb.tsv, which  |
//| python -m src.fx_ny_orb.export rebuilds after every live pass.    |
//| Python computes every price, time and text; this file only draws. |
//| It never places, modifies or closes an order.                     |
//|                                                                   |
//| Per day: the opening-range box (14:00-15:00 Berlin labels = the   |
//| chart's server time) and its high/low lines to 21:00, the         |
//| forecast band (size, not direction) to 22:00, red on a hot day,   |
//| the realised session range after settlement, and past simulated   |
//| trades. A live entry arrow with SL/TP exists only when the file   |
//| carries one: the rule's historical cell must be KEEP and today    |
//| hot (the gate is enforced in Python).                             |
//+------------------------------------------------------------------+
#property copyright   "EUR/USD Prophet"
#property version     "1.00"
#property description "NY opening range, range forecast and the breakout rule (display only)."
#property description "Research - live sign only after a historical KEEP; not a trading result."
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

#define NYO_FILE   "EURUSDProphet\\ny_orb.tsv"
#define NYO_PREFIX "NYORB_"

enum ENUM_NYO_LANG { NYO_BG = 0, NYO_EN = 1 };

input ENUM_NYO_LANG    InpLanguage    = NYO_BG;            // Език / Language
input bool             InpShowNonHot  = false;             // Show simulated trades of non-hot days
input color            InpRangeColor  = clrSteelBlue;      // Opening range
input color            InpHotColor    = clrRed;            // Forecast band, hot day
input color            InpColdColor   = clrDarkGray;       // Forecast band, ordinary day
input color            InpRealColor   = clrGoldenrod;      // Realised session range
input color            InpWinColor    = clrLimeGreen;      // Trade with net R > 0
input color            InpLossColor   = clrTomato;         // Trade with net R <= 0
input color            InpLiveColor   = clrDodgerBlue;     // Live rule signal
input ENUM_BASE_CORNER InpCorner      = CORNER_LEFT_UPPER; // Status corner
input int              InpFontSize    = 8;                 // Font size

datetime g_file_mod = 0;
string   g_status   = "";
string   g_gate_msg = "";
bool     g_gate     = false;

//+------------------------------------------------------------------+
string Pick(const string bg, const string en)
  {
   return (InpLanguage == NYO_BG) ? bg : en;
  }

bool ChartPairMatches(const string pair)
  {
   return StringFind(_Symbol, pair) == 0;
  }

bool TimeframeShown()
  {
   ENUM_TIMEFRAMES tf = (ENUM_TIMEFRAMES)Period();
   return tf == PERIOD_M1 || tf == PERIOD_M5 || tf == PERIOD_M15 || tf == PERIOD_M30 || tf == PERIOD_H1;
  }

//+------------------------------------------------------------------+
void Rect(const string name, datetime t1, double p1, datetime t2, double p2, color c, bool fill, ENUM_LINE_STYLE st)
  {
   ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_FILL, fill);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_STYLE, st);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

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

void Arrow(const string name, datetime t, double p, int code, color c, int w, long anchor)
  {
   ObjectCreate(0, name, OBJ_ARROW, 0, t, p);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE, code);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, w);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, anchor);
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
//| Column index by header name; -1 when absent.                     |
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

//+------------------------------------------------------------------+
//| Draw one day row.                                                |
//+------------------------------------------------------------------+
void DrawRow(string &f[], string &head[], int n)
  {
   if(Get(f, head, "status") != "ok" || !ChartPairMatches(Get(f, head, "pair")))
      return;
   string id = NYO_PREFIX + Get(f, head, "pair") + "_" + Get(f, head, "date") + "_";
   bool hot = Get(f, head, "hot") == "1";
   datetime or0 = StringToTime(Get(f, head, "or_start"));
   datetime or1 = StringToTime(Get(f, head, "or_end"));
   datetime lend = StringToTime(Get(f, head, "line_end"));
   datetime b0 = StringToTime(Get(f, head, "band_start"));
   datetime b1 = StringToTime(Get(f, head, "band_end"));
   double orh = StringToDouble(Get(f, head, "or_high"));
   double orl = StringToDouble(Get(f, head, "or_low"));
   double blo = StringToDouble(Get(f, head, "band_lo"));
   double bhi = StringToDouble(Get(f, head, "band_hi"));

   // opening range box and its high / low lines to 21:00
   Rect(id + "or", or0, orh, or1, orl, InpRangeColor, true, STYLE_SOLID);
   Line(id + "orh", or1, orh, lend, orh, InpRangeColor, STYLE_SOLID, 1);
   Line(id + "orl", or1, orl, lend, orl, InpRangeColor, STYLE_SOLID, 1);

   // forecast band: size, not direction
   color bc = hot ? InpHotColor : InpColdColor;
   Line(id + "bhi", b0, bhi, b1, bhi, bc, STYLE_DASH, hot ? 2 : 1);
   Line(id + "blo", b0, blo, b1, blo, bc, STYLE_DASH, hot ? 2 : 1);
   Text(id + "blab", b0, bhi, Pick(Get(f, head, "band_label_bg"), Get(f, head, "band_label_en")), bc);

   // realised session range once settled
   string rh = Get(f, head, "real_high"), rl = Get(f, head, "real_low");
   if(rh != "" && rl != "")
      Rect(id + "real", b0, StringToDouble(rh), b1, StringToDouble(rl), InpRealColor, false, STYLE_DOT);

   // trades
   string kind = Get(f, head, "kind");
   if(kind == "")
      return;
   if(kind == "sim" && !hot && !InpShowNonHot)
      return;
   int dir = (int)StringToInteger(Get(f, head, "direction"));
   datetime te = StringToTime(Get(f, head, "entry_time"));
   double pe = StringToDouble(Get(f, head, "entry"));
   double ps = StringToDouble(Get(f, head, "stop"));
   double pt = StringToDouble(Get(f, head, "target"));
   string lab = Pick(Get(f, head, "label_bg"), Get(f, head, "label_en"));
   if(kind == "live")
     {
      Arrow(id + "in", te, pe, dir > 0 ? 233 : 234, InpLiveColor, 3, dir > 0 ? ANCHOR_TOP : ANCHOR_BOTTOM);
      Line(id + "sl", te, ps, lend, ps, InpLossColor, STYLE_DASH, 1);
      Line(id + "tp", te, pt, lend, pt, InpWinColor, STYLE_DASH, 1);
      Text(id + "lab", te, pe, lab, InpLiveColor);
      return;
     }
   double r = StringToDouble(Get(f, head, "net_r"));
   color tc = (r > 0) ? InpWinColor : InpLossColor;
   datetime tx = StringToTime(Get(f, head, "exit_time"));
   double px = StringToDouble(Get(f, head, "exit"));
   Arrow(id + "in", te, pe, dir > 0 ? 233 : 234, tc, 2, dir > 0 ? ANCHOR_TOP : ANCHOR_BOTTOM);
   Line(id + "sl", te, ps, tx, ps, InpLossColor, STYLE_DOT, 1);
   Line(id + "tp", te, pt, tx, pt, InpWinColor, STYLE_DOT, 1);
   Line(id + "path", te, pe, tx, px, tc, STYLE_SOLID, 1);
   Arrow(id + "out", tx, px, 251, tc, 2, ANCHOR_CENTER);
   Text(id + "lab", te, pe, lab, tc);
  }

//+------------------------------------------------------------------+
//| Read the file when it changed and redraw everything.             |
//+------------------------------------------------------------------+
void Refresh(bool force)
  {
   if(!FileIsExist(NYO_FILE, FILE_COMMON))
     {
      ObjectsDeleteAll(0, NYO_PREFIX);
      CornerLabel(NYO_PREFIX + "status0", 0, Pick("Няма файл ny_orb.tsv (пуснете логера).",
                                                   "No ny_orb.tsv yet (run the logger)."), clrSilver);
      ChartRedraw();
      return;
     }
   datetime mod = (datetime)FileGetInteger(NYO_FILE, FILE_MODIFY_DATE, true);
   if(!force && mod == g_file_mod)
      return;
   int h = FileOpen(NYO_FILE, FILE_READ | FILE_BIN | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON);
   if(h == INVALID_HANDLE)
      return;                                   // keep what is drawn; retry on the next timer
   uchar bytes[];
   uint got = FileReadArray(h, bytes);
   FileClose(h);
   g_file_mod = mod;
   string text = CharArrayToString(bytes, 0, (int)got, CP_UTF8);
   string lines[];
   int nl = StringSplit(text, '\n', lines);

   ObjectsDeleteAll(0, NYO_PREFIX);
   g_status = "";
   g_gate_msg = "";
   g_gate = false;
   string head[];
   int nh = 0, rows = 0;
   for(int i = 0; i < nl; i++)
     {
      string ln = lines[i];
      StringReplace(ln, "\r", "");
      if(ln == "")
         continue;
      string f[];
      int nf = StringSplit(ln, '\t', f);
      if(f[0] == "#meta")
        {
         for(int k = 1; k < nf; k++)
            if(f[k] == "gate=open")
               g_gate = true;
         continue;
        }
      if(f[0] == "#text")
        {
         if(nf >= 4 && f[1] == "status")
            g_status = Pick(f[2], f[3]);
         if(nf >= 4 && f[1] == "gate_closed")
            g_gate_msg = Pick(f[2], f[3]);
         continue;
        }
      if(nh == 0)
        {
         nh = StringSplit(ln, '\t', head);
         continue;
        }
      if(!TimeframeShown())
         continue;
      DrawRow(f, head, nf);
      rows++;
     }
   CornerLabel(NYO_PREFIX + "status0", 0, g_status, clrSilver);
   if(!g_gate && g_gate_msg != "")
      CornerLabel(NYO_PREFIX + "status1", 1, g_gate_msg, clrSilver);
   if(!TimeframeShown())
      CornerLabel(NYO_PREFIX + "status2", 2, Pick("Рисува се само на M1-H1.", "Drawn on M1-H1 only."), clrSilver);
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
   ObjectsDeleteAll(0, NYO_PREFIX);
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
