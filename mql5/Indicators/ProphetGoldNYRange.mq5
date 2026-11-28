//+------------------------------------------------------------------+
//| ProphetGoldNYRange.mq5                                            |
//| EUR/USD Prophet - the expected size of gold's New York session    |
//| (openspec change gold-nyrange-forward, spec                       |
//| gold-nyrange-forward/chart-display).                              |
//|                                                                   |
//| DISPLAY ONLY. Reads Common\Files\EURUSDProphet\gold_ny_range.tsv, |
//| which python -m src.gold_nyfwd.export rebuilds after every pass.  |
//| Python computes every price, time and text; this file only draws. |
//| It never places, modifies or closes an order.                     |
//|                                                                   |
//| Per day: a band of total height F centred on the 14:00 close,     |
//| from 15:00 to 22:00 Berlin labels (the chart's server time), and  |
//| after settlement a dotted box of the realised high and low.       |
//| Size only, never a direction. Drawn on GOLD at H1 and below.      |
//+------------------------------------------------------------------+
#property copyright   "EUR/USD Prophet"
#property version     "1.00"
#property description "Expected size of gold's New York session (display only)."
#property description "Research forward test - size, not direction; no verdict before scoring."
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

#define GNR_FILE   "EURUSDProphet\\gold_ny_range.tsv"
#define GNR_PREFIX "GNYR_"
#define GNR_SYMBOL "GOLD"

enum ENUM_GNR_LANG { GNR_BG = 0, GNR_EN = 1 };

input ENUM_GNR_LANG    InpLanguage   = GNR_BG;            // Език / Language
input color            InpBandColor  = clrDarkOrange;     // Expected size band
input color            InpRealColor  = clrGoldenrod;      // Realised session range
input ENUM_BASE_CORNER InpCorner     = CORNER_LEFT_UPPER; // Status corner
input int              InpFontSize   = 8;                 // Font size

datetime g_file_mod = 0;

//+------------------------------------------------------------------+
string Pick(const string bg, const string en)
  {
   return (InpLanguage == GNR_BG) ? bg : en;
  }

bool SymbolShown()
  {
   return StringFind(_Symbol, GNR_SYMBOL) == 0;
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

//+------------------------------------------------------------------+
//| Draw one day: the size band and, once settled, the realised box. |
//+------------------------------------------------------------------+
void DrawRow(string &f[], string &head[])
  {
   if(Get(f, head, "status") != "ok")
      return;
   string id = GNR_PREFIX + Get(f, head, "date") + "_";
   datetime b0 = StringToTime(Get(f, head, "band_start"));
   datetime b1 = StringToTime(Get(f, head, "band_end"));
   double blo = StringToDouble(Get(f, head, "band_lo"));
   double bhi = StringToDouble(Get(f, head, "band_hi"));
   double c14 = StringToDouble(Get(f, head, "close14"));

   Rect(id + "band", b0, bhi, b1, blo, InpBandColor, false, STYLE_DASH);
   Line(id + "mid", b0, c14, b1, c14, InpBandColor, STYLE_DOT, 1);
   Text(id + "lab", b0, bhi, Pick(Get(f, head, "label_bg"), Get(f, head, "label_en")), InpBandColor);

   string rh = Get(f, head, "real_high"), rl = Get(f, head, "real_low");
   if(rh != "" && rl != "")
      Rect(id + "real", b0, StringToDouble(rh), b1, StringToDouble(rl), InpRealColor, false, STYLE_DOT);
  }

//+------------------------------------------------------------------+
//| Read the file when it changed and redraw everything.             |
//+------------------------------------------------------------------+
void Refresh(bool force)
  {
   if(!SymbolShown())
     {
      ObjectsDeleteAll(0, GNR_PREFIX);
      CornerLabel(GNR_PREFIX + "status0", 0, Pick("Този индикатор е само за GOLD.", "This indicator is for GOLD only."),
                  clrSilver);
      ChartRedraw();
      return;
     }
   if(!FileIsExist(GNR_FILE, FILE_COMMON))
     {
      ObjectsDeleteAll(0, GNR_PREFIX);
      CornerLabel(GNR_PREFIX + "status0", 0, Pick("Няма файл gold_ny_range.tsv (пуснете прогнозата).",
                                                   "No gold_ny_range.tsv yet (run the forecast pass)."), clrSilver);
      ChartRedraw();
      return;
     }
   datetime mod = (datetime)FileGetInteger(GNR_FILE, FILE_MODIFY_DATE, true);
   if(!force && mod == g_file_mod)
      return;
   int h = FileOpen(GNR_FILE, FILE_READ | FILE_BIN | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON);
   if(h == INVALID_HANDLE)
      return;                                   // keep what is drawn; retry on the next timer
   uchar bytes[];
   uint got = FileReadArray(h, bytes);
   FileClose(h);
   g_file_mod = mod;
   string text = CharArrayToString(bytes, 0, (int)got, CP_UTF8);
   string lines[];
   int nl = StringSplit(text, '\n', lines);

   ObjectsDeleteAll(0, GNR_PREFIX);
   string status = "", note = "";
   string head[];
   int nh = 0;
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
         continue;
        }
      if(nh == 0)
        {
         nh = StringSplit(ln, '\t', head);
         continue;
        }
      if(TimeframeShown())
         DrawRow(f, head);
     }
   CornerLabel(GNR_PREFIX + "status0", 0, status, clrSilver);
   CornerLabel(GNR_PREFIX + "status1", 1, note, clrSilver);
   if(!TimeframeShown())
      CornerLabel(GNR_PREFIX + "status2", 2, Pick("Рисува се само на M1-H1.", "Drawn on M1-H1 only."), clrSilver);
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
   ObjectsDeleteAll(0, GNR_PREFIX);
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
