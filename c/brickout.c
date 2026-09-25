/*
 * brickout: a self-playing round brick-breaker arcade game on the iCUE LINK AIO pump
 * LCD, played by your two GPUs.
 *
 * The screen is the playfield: a pixel-art picture made of bricks sits in the middle
 * and two curved paddles run around the rim, blue on the left for GPU 0's vLLM server,
 * orange on the right for GPU 1's. Each server keeps one ball in play per ~170 tok/s
 * (up to 6), and balls fly faster as its tok/s rises; clear the picture and the next
 * stage drops in. Power-up capsules split balls or widen a paddle. When nothing is
 * generating the cabinet falls back to attract mode: one white ball, INSERT COIN, and
 * the session's token count as the HI-SCORE.
 * Run with --demo to simulate data, --showcase for a scripted loop, --bench to write
 * preview PNGs.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <cairo/cairo.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/hidraw.h>
#include <math.h>
#include <netinet/in.h>
#include <nvml.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <turbojpeg.h>
#include <unistd.h>

#define SIZE            480
#define FPS             24
#define REPORT_SIZE     1024
#define HEADER_SIZE     8
#define CHUNK_SIZE      (REPORT_SIZE - HEADER_SIZE)
#define JPEG_QUALITY    85
#define POWER_MAX       1150.0      /* both cards near their 575 W limits */
#define N_GPUS          2


static const char  *gpu_bus[N_GPUS]  = { "00000000:01:00.0", "00000000:03:00.0" };  /* ZOTAC, TUF */
static const int    vllm_ports[]     = { 18090, 18091 };
#define N_PORTS     (int)(sizeof(vllm_ports) / sizeof(vllm_ports[0]))

typedef struct { double r, g, b; } rgb;

__attribute__((unused)) static const rgb BLUE   = { 61 / 255.0, 174 / 255.0, 233 / 255.0 };
__attribute__((unused)) static const rgb ORANGE = { 233 / 255.0, 120 / 255.0, 61 / 255.0 };
__attribute__((unused)) static const rgb WHITE  = { 240 / 255.0, 240 / 255.0, 245 / 255.0 };
__attribute__((unused)) static const rgb DIM    = { 110 / 255.0, 115 / 255.0, 125 / 255.0 };
__attribute__((unused)) static const rgb TRACK  = { 38 / 255.0, 40 / 255.0, 48 / 255.0 };
__attribute__((unused)) static const rgb BG     = { 6 / 255.0, 7 / 255.0, 12 / 255.0 };

typedef struct {
    double load[N_GPUS];        /* 0..1 */
    double power[N_GPUS];       /* W */
    int    temp[N_GPUS];        /* C */
    double tok_s;               /* all servers */
    double tok_port[2];         /* per server: [0] ZOTAC's vLLM, [1] TUF's vLLM */
    int    running;
    int    running_port[2];     /* running requests per server */
} stats;

static volatile sig_atomic_t stop;
static int demo;

/*
 * Data source. By default the display is driven by vLLM tokens/sec. With
 * LLM_REACTOR_SOURCE=gpu in the environment (or --gpu-load) it is driven by GPU
 * activity instead: each GPU's load is mapped onto the same range the display
 * expects (GPU_FULL_RATE "tok/s" at 100%), so it works for any GPU workload, and the
 * text shows GPU % instead of tok/s. Activity is half utilisation, half power draw
 * between GPU_IDLE_W and GPU_MAX_W: utilisation alone can sit at 100% while the card
 * is barely working, the watts show how hard it really is.
 */
#define GPU_FULL_RATE   900.0
#define GPU_IDLE_W      40.0        /* board power at idle */
#define GPU_MAX_W       575.0       /* board power limit */
static int gpu_source;

/* Numbers as shown on screen: tok/s, or GPU activity % in GPU mode */
__attribute__((unused)) static double shown_rate(double tok)        /* totals: average % */
{
    return gpu_source ? fmin(100, tok / (N_GPUS * GPU_FULL_RATE) * 100) : tok;
}
__attribute__((unused)) static double shown_gpu_rate(double tok)    /* one GPU */
{
    return gpu_source ? fmin(100, tok / GPU_FULL_RATE * 100) : tok;
}

/* Units to match: on their own, upper case, spelled out, and right after a number */
__attribute__((unused)) static const char *rate_unit(void) { return gpu_source ? "% GPU" : "tok/s"; }
__attribute__((unused)) static const char *rate_unit_uc(void) { return gpu_source ? "% GPU" : "TOK/S"; }
__attribute__((unused)) static const char *rate_unit_long(void) { return gpu_source ? "GPU LOAD %" : "TOKENS / SEC"; }
__attribute__((unused)) static const char *rate_suffix(void) { return gpu_source ? "% GPU" : " tok/s"; }

static void on_signal(int sig) { (void)sig; stop = 1; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double clamp01(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

static rgb lerp(rgb a, rgb b, double t)
{
    rgb c = { a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t };
    return c;
}

/* cool cyan -> amber -> hot red as power rises */
__attribute__((unused)) static rgb heat_color(double t)
{
    static const rgb c0 = { 40 / 255.0, 200 / 255.0, 1.0 };
    static const rgb c1 = { 1.0, 170 / 255.0, 40 / 255.0 };
    static const rgb c2 = { 1.0, 40 / 255.0, 30 / 255.0 };
    t = clamp01(t);
    return t <= 0.55 ? lerp(c0, c1, t / 0.55) : lerp(c1, c2, (t - 0.55) / 0.45);
}

static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

/* ---------------------------------------------------------------- LCD */

/*
 * Set LCD_DUMP_DIR to write every frame as a JPEG into that directory instead of
 * sending it to the pump (used to make the README GIFs; no device needed).
 */
static const char *lcd_dump_dir(void)
{
    const char *d = getenv("LCD_DUMP_DIR");
    return d && *d ? d : NULL;
}

static int lcd_open(void)
{
    if (lcd_dump_dir())
        return open("/dev/null", O_WRONLY | O_CLOEXEC);

    DIR *dir = opendir("/sys/class/hidraw");
    struct dirent *e;
    char path[512], buf[1024];
    int fd = -1;

    if (!dir)
        return -1;
    while ((e = readdir(dir)) && fd < 0) {
        if (strncmp(e->d_name, "hidraw", 6) != 0)
            continue;
        snprintf(path, sizeof(path), "/sys/class/hidraw/%s/device/uevent", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        buf[n] = 0;
        if (strcasestr(buf, "00001B1C:00000C4E")) {
            snprintf(path, sizeof(path), "/dev/%s", e->d_name);
            fd = open(path, O_RDWR | O_CLOEXEC);
        }
    }
    closedir(dir);
    return fd;
}

static void lcd_brightness(int fd, int percent)
{
    if (lcd_dump_dir())
        return;
    unsigned char rep[4] = { 0x03, 0x0B, (unsigned char)percent, 0x01 };
    ioctl(fd, HIDIOCSFEATURE(sizeof(rep)), rep);
}

static int lcd_send(int fd, const unsigned char *jpeg, unsigned long len)
{
    if (lcd_dump_dir()) {
        static int frame;
        char path[512];
        snprintf(path, sizeof(path), "%s/frame_%05d.jpg", lcd_dump_dir(), frame++);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(jpeg, 1, len, f);
            fclose(f);
        }
        return 0;
    }

    unsigned char rep[REPORT_SIZE];
    unsigned long off = 0;
    int idx = 0;

    while (off < len) {
        unsigned long n = len - off > CHUNK_SIZE ? CHUNK_SIZE : len - off;
        memset(rep, 0, sizeof(rep));
        rep[0] = 0x02;
        rep[1] = 0x05;
        rep[2] = 0x01;
        rep[3] = (off + n == len) ? 0x01 : 0x00;   /* last chunk tells the panel to render */
        rep[4] = (unsigned char)idx++;
        rep[6] = n & 0xFF;
        rep[7] = (n >> 8) & 0xFF;
        memcpy(rep + HEADER_SIZE, jpeg + off, n);
        if (write(fd, rep, sizeof(rep)) != (ssize_t)sizeof(rep))
            return -1;
        off += n;
    }
    return 0;
}

/* ---------------------------------------------------------------- data */

static nvmlDevice_t nvml_dev[N_GPUS];
static int nvml_ok;

static void gpus_init(void)
{
    if (nvmlInit_v2() != NVML_SUCCESS)
        return;
    nvml_ok = 1;
    for (int i = 0; i < N_GPUS; i++)
        if (nvmlDeviceGetHandleByPciBusId_v2(gpu_bus[i], &nvml_dev[i]) != NVML_SUCCESS)
            nvml_dev[i] = NULL;
}

static void gpus_poll(stats *s)
{
    if (!nvml_ok)
        return;
    for (int i = 0; i < N_GPUS; i++) {
        nvmlUtilization_t u;
        nvmlTemperature_t temp = { .version = nvmlTemperature_v1, .sensorType = NVML_TEMPERATURE_GPU };
        unsigned int mw;
        if (!nvml_dev[i])
            continue;
        if (nvmlDeviceGetUtilizationRates(nvml_dev[i], &u) == NVML_SUCCESS)
            s->load[i] = u.gpu / 100.0;
        if (nvmlDeviceGetPowerUsage(nvml_dev[i], &mw) == NVML_SUCCESS)
            s->power[i] = mw / 1000.0;
        if (nvmlDeviceGetTemperatureV(nvml_dev[i], &temp) == NVML_SUCCESS)
            s->temp[i] = temp.temperature;
    }
}

/* Fetch http://127.0.0.1:port/metrics into buf. Returns bytes read or -1. */
static long http_metrics(int port, char *buf, long cap)
{
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(port) };
    struct timeval tv = { .tv_sec = 0, .tv_usec = 300000 };
    static const char req[] = "GET /metrics HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n";
    long got = 0;
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);

    if (fd < 0)
        return -1;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
        write(fd, req, sizeof(req) - 1) != (ssize_t)(sizeof(req) - 1)) {
        close(fd);
        return -1;
    }
    while (got < cap - 1) {
        ssize_t n = read(fd, buf + got, cap - 1 - got);
        if (n <= 0)
            break;
        got += n;
    }
    close(fd);
    buf[got] = 0;
    return got;
}

/* Sum every sample of a metric family, e.g. "vllm:generation_tokens_total". */
static double metric_sum(const char *text, const char *name)
{
    size_t nlen = strlen(name);
    double sum = 0;
    const char *p = text;

    while ((p = strstr(p, name))) {
        if ((p == text || p[-1] == '\n') && (p[nlen] == '{' || p[nlen] == ' ')) {
            const char *eol = strchr(p, '\n');
            const char *sp = p + nlen;
            if (*sp == '{')
                sp = strchr(sp, '}');
            if (sp && (!eol || sp < eol))
                sum += strtod(sp + 1, NULL);
        }
        p += nlen;
    }
    return sum;
}

static void vllm_poll(stats *s, double t)
{
    static char *buf;
    static double last_tokens[N_PORTS], last_t[N_PORTS];
    static int have[N_PORTS];
    int running = 0;

    if (!buf && !(buf = malloc(1 << 20)))
        return;
    s->tok_s = 0;
    for (int i = 0; i < N_PORTS; i++) {
        double tokens, rate = 0;
        if (http_metrics(vllm_ports[i], buf, 1 << 20) <= 0) {
            have[i] = 0;
            s->running_port[i] = 0;
            s->tok_port[i] *= 0.5;
            s->tok_s += s->tok_port[i];
            continue;
        }
        tokens   = metric_sum(buf, "vllm:generation_tokens_total");
        s->running_port[i] = (int)metric_sum(buf, "vllm:num_requests_running");
        running += s->running_port[i];
        if (have[i] && tokens >= last_tokens[i] && t > last_t[i])
            rate = (tokens - last_tokens[i]) / (t - last_t[i]);
        s->tok_port[i] = 0.6 * s->tok_port[i] + 0.4 * rate;
        s->tok_s += s->tok_port[i];
        last_tokens[i] = tokens;
        last_t[i] = t;
        have[i] = 1;
    }
    s->running = running;
}

/*
 * GPU mode: turn each GPU's activity (see GPU_FULL_RATE) into a token rate and a
 * running-request count, so the display's thresholds and idle detection just work.
 */
static void gpu_rate_poll(stats *s)
{
    s->tok_s = 0;
    s->running = 0;
    for (int i = 0; i < N_GPUS; i++) {
        double pw = clamp01((s->power[i] - GPU_IDLE_W) / (GPU_MAX_W - GPU_IDLE_W));
        double a = 0.5 * clamp01(s->load[i]) + 0.5 * pw;
        if (a < 0.03)                   /* no idle jitter */
            a = 0;
        s->tok_port[i] = a * GPU_FULL_RATE;
        s->running_port[i] = a > 0.05 ? (int)ceil(a * 4) : 0;
        s->tok_s += a * GPU_FULL_RATE;
        s->running += a > 0.05;
    }
}

static void demo_poll(stats *s, double t)
{
    double busy = 0.5 + 0.5 * sin(t * 0.25);
    s->load[0]  = clamp01(busy + (rand() / (double)RAND_MAX - 0.5) * 0.1);
    s->load[1]  = clamp01(busy * 0.9 + (rand() / (double)RAND_MAX - 0.5) * 0.1);
    s->power[0] = 25 + 420 * s->load[0];
    s->power[1] = 30 + 410 * s->load[1];
    s->temp[0]  = (int)(35 + 20 * busy);
    s->temp[1]  = (int)(42 + 30 * busy);
    s->tok_port[0] = 0.8 * s->tok_port[0] + 0.2 * (busy > 0.15 ? 140 * busy : 0);
    s->tok_port[1] = 0.8 * s->tok_port[1] + 0.2 * (busy > 0.25 ? 120 * (busy - 0.1) : 0);
    s->tok_s    = s->tok_port[0] + s->tok_port[1];
    s->running  = (int)(busy * 6);
}


/* ---------------------------------------------------------------- scene */

/*
 * The playfield background, the CRT scanlines and the cabinet bezel are drawn once and
 * cached, the bricks are re-cached only when one breaks, and the text layer is redrawn
 * only when its content changes (numbers update 4x a second). Each frame draws the
 * balls, paddles, particles and bezel bulbs on top, and the frame rate drops when idle.
 */
#define FPS_BUSY        20
#define FPS_IDLE        15
#define TOK_PER_BALL    170.0       /* one ball in play per this many tok/s per server */
#define MAX_OWN_BALLS   6           /* per server, not counting power-up extras */
#define MAX_BALLS       40
#define MAX_PARTS       400
#define MAX_CAPS        6
#define CX              240.0
#define CY              240.0
#define R_PAD           207.0       /* inner face of the paddles */
#define PAD_T           13.0        /* paddle thickness */
#define PAD_HW          0.25        /* paddle half-width, radians */
#define BALL_R          7.0
#define R_HIT           (R_PAD - BALL_R)
#define GW              13          /* brick grid */
#define GH              11
#define CW              18.0        /* brick size */
#define CH              14.0
#define GRID_X0         (CX - GW * CW / 2.0)
#define GRID_Y0         (CY - GH * CH / 2.0 - 6)
#define N_BULBS         30

static double frand(void) { return rand() / (double)RAND_MAX; }

static double wrap_pi(double a)
{
    while (a > M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
}

/* ------------------------------------------------ 5x7 pixel font */

static const char *font[128][7] = {
    ['0'] = { " ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### " },
    ['1'] = { "  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### " },
    ['2'] = { " ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####" },
    ['3'] = { "#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### " },
    ['4'] = { "   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # " },
    ['5'] = { "#####", "#    ", "#### ", "    #", "    #", "#   #", " ### " },
    ['6'] = { "  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### " },
    ['7'] = { "#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   " },
    ['8'] = { " ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### " },
    ['9'] = { " ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  " },
    ['A'] = { " ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #" },
    ['B'] = { "#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### " },
    ['C'] = { " ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### " },
    ['D'] = { "#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### " },
    ['E'] = { "#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####" },
    ['F'] = { "#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    " },
    ['G'] = { " ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ####" },
    ['H'] = { "#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #" },
    ['I'] = { " ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### " },
    ['J'] = { "  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  " },
    ['K'] = { "#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #" },
    ['L'] = { "#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####" },
    ['M'] = { "#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #" },
    ['N'] = { "#   #", "#   #", "##  #", "# # #", "#  ##", "#   #", "#   #" },
    ['O'] = { " ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### " },
    ['P'] = { "#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    " },
    ['Q'] = { " ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #" },
    ['R'] = { "#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #" },
    ['S'] = { " ####", "#    ", "#    ", " ### ", "    #", "    #", "#### " },
    ['T'] = { "#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  " },
    ['U'] = { "#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### " },
    ['V'] = { "#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  " },
    ['W'] = { "#   #", "#   #", "#   #", "# # #", "# # #", "# # #", " # # " },
    ['X'] = { "#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #" },
    ['Y'] = { "#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  " },
    ['Z'] = { "#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####" },
    ['/'] = { "    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    " },
    ['-'] = { "     ", "     ", "     ", "#####", "     ", "     ", "     " },
    ['+'] = { "     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     " },
    ['!'] = { "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "     ", "  #  " },
    ['.'] = { "     ", "     ", "     ", "     ", "     ", " ##  ", " ##  " },
    ['%'] = { "##  #", "##  #", "   # ", "  #  ", " #   ", "#  ##", "#  ##" },
};

static double ptext_width(const char *s, double sc)
{
    size_t n = strlen(s);
    return n ? (n * 6 - 1) * sc : 0;
}

static void ptext_path(cairo_t *cr, double x0, double y, double sc, double grow, const char *s)
{
    for (const char *p = s; *p; p++, x0 += 6 * sc) {
        const char **g = font[(unsigned char)*p & 127];
        if (!g[0])
            continue;
        for (int r = 0; r < 7; r++)
            for (int c = 0; c < 5; c++)
                if (g[r][c] == '#')
                    cairo_rectangle(cr, x0 + c * sc - grow, y + r * sc - grow, sc + 2 * grow, sc + 2 * grow);
    }
}

/* Chunky arcade text centred on cx, top at y: black outline, then a top-lit gradient */
static void ptext(cairo_t *cr, double cx, double y, double sc, rgb col, const char *s)
{
    double x0 = floor(cx - ptext_width(s, sc) / 2), o = fmax(2, sc * 0.55);
    cairo_new_path(cr);
    ptext_path(cr, x0, y, sc, o, s);
    cairo_set_source_rgba(cr, 0, 0, 0, 0.92);
    cairo_fill(cr);
    ptext_path(cr, x0, y, sc, 0, s);
    rgb hi = lerp(col, WHITE, 0.45), lo = lerp(col, (rgb){ 0, 0, 0 }, 0.3);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, y, 0, y + 7 * sc);
    cairo_pattern_add_color_stop_rgb(g, 0, hi.r, hi.g, hi.b);
    cairo_pattern_add_color_stop_rgb(g, 0.45, col.r, col.g, col.b);
    cairo_pattern_add_color_stop_rgb(g, 1, lo.r, lo.g, lo.b);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
}

/* ------------------------------------------------ stages: pictures made of bricks */

static const char *stage_art[][GH + 1] = {
    { "..#.....#..",            /* invader */
      "...#...#...",
      "..#######..",
      ".##.###.##.",
      "###########",
      "#.#######.#",
      "#.#.....#.#",
      "...##.##...", NULL },
    { "...ppppp...",            /* ghost */
      ".ppppppppp.",
      "ppwwppppwwp",
      "pwwbbppwwbb",
      "pwwbbppwwbb",
      "ppwwppppwwp",
      "ppppppppppp",
      "ppppppppppp",
      "ppppppppppp",
      "pp.pp.pp.pp", NULL },
    { ".rrr...rrr.",            /* heart */
      "rrrrr.rrrrr",
      "rryrrrrrrrr",
      "ryrrrrrrrrr",
      "rrrrrrrrrrr",
      ".rrrrrrrrr.",
      "..rrrrrrr..",
      "...rrrrr...",
      "....rrr....",
      ".....r.....", NULL },
    { "...sssss...",            /* skull, silver bricks take two hits */
      ".sssssssss.",
      "sssssssssss",
      "ss...s...ss",
      "ss...s...ss",
      "sssssssssss",
      "sssss.sssss",
      ".sssssssss.",
      "..s.s.s.s..",
      "..sssssss..", NULL },
    { "....ccccc....",          /* flying saucer */
      "...ccwwwcc...",
      "..ccccccccc..",
      ".ppppppppppp.",
      "ppyppyppyppyp",
      ".ppppppppppp.",
      "...ggg.ggg...",
      "..gg.....gg..", NULL },
    { "....#...#....",          /* squid invader */
      "...#######...",
      "..##.###.##..",
      ".###########.",
      ".#.#######.#.",
      ".#.#.....#.#.",
      "....##.##....",
      "..##.....##..", NULL },
};
#define N_STAGES (int)(sizeof(stage_art) / sizeof(stage_art[0]))

static rgb brick_rgb(char ch, int row)
{
    static const rgb rainbow[8] = {
        { 0.95, 0.24, 0.26 }, { 1.00, 0.55, 0.14 }, { 1.00, 0.85, 0.20 }, { 0.36, 0.90, 0.34 },
        { 0.20, 0.84, 0.95 }, { 0.30, 0.48, 1.00 }, { 0.70, 0.40, 1.00 }, { 1.00, 0.42, 0.78 },
    };
    switch (ch) {
    case 'r': return (rgb){ 0.93, 0.20, 0.24 };
    case 'o': return (rgb){ 1.00, 0.55, 0.14 };
    case 'y': return (rgb){ 1.00, 0.88, 0.26 };
    case 'g': return (rgb){ 0.36, 0.90, 0.34 };
    case 'c': return (rgb){ 0.22, 0.85, 0.95 };
    case 'b': return (rgb){ 0.20, 0.36, 0.98 };
    case 'p': return (rgb){ 1.00, 0.46, 0.80 };
    case 'w': return (rgb){ 0.96, 0.96, 1.00 };
    case 's': return (rgb){ 0.74, 0.77, 0.84 };
    default:  return rainbow[row % 8];
    }
}

typedef struct {
    int    hp;                  /* 0 = empty */
    int    silver;
    double delay;               /* drop-in delay during a stage build */
    rgb    col;
} brick;

typedef struct {
    double x, y, vx, vy, speed, aim;
    double tx[3], ty[3];        /* trail */
    int    owner;               /* 0 = GPU 0 (blue), 1 = GPU 1 (orange), 2 = attract mode (white) */
    int    alive, lost, bonus;
} ball;

typedef struct {
    double a, va, hw, flash, wide_t;
} paddle;

typedef struct {
    double x, y, vx, vy, life, max, size;
    rgb    col;
    int    flash;               /* 1 = brick-sized white flash */
} particle;

typedef struct {
    double x, y, vx, vy;
    int    type, alive;         /* 0 = M (multiball), 1 = W (wide paddle) */
} capsule;

enum { PLAY, CLEAR, BUILD };

static brick    grid[GH][GW];
static ball     balls[MAX_BALLS];
static paddle   pads[2];
static particle parts[MAX_PARTS];
static capsule  caps[MAX_CAPS];
static int      stage = 1, phase = BUILD, bricks_left, bricks_dirty = 1, serve_alt;
static double   phase_t, serve_cd[3], bulb_phase, score;
static int      misses;
static cairo_surface_t *bg_cache, *overlay_cache, *brick_cache, *text_cache, *glow[3];

static const rgb OWNER_COL[3] = {
    { 61 / 255.0, 174 / 255.0, 233 / 255.0 },
    { 243 / 255.0, 130 / 255.0, 61 / 255.0 },
    { 0.96, 0.96, 0.92 },
};

static void load_stage(int n)
{
    const char **art = stage_art[(n - 1) % N_STAGES];
    int h = 0, w = (int)strlen(art[0]);
    while (art[h])
        h++;
    int ox = (GW - w) / 2, oy = (GH - h) / 2;
    memset(grid, 0, sizeof(grid));
    bricks_left = 0;
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            char ch = art[r][c];
            if (ch == '.')
                continue;
            brick *b = &grid[oy + r][ox + c];
            b->silver = ch == 's';
            b->hp = b->silver ? 2 : 1;
            b->col = brick_rgb(ch, r);
            b->delay = (h - 1 - r) * 0.07 + fabs(c - (w - 1) / 2.0) * 0.015;
            bricks_left++;
        }
    phase = BUILD;
    phase_t = 0;
    bricks_dirty = 1;
}

#define BUILD_TIME  1.6
#define CLEAR_TIME  1.4

static int brick_solid(int r, int c)
{
    if (r < 0 || r >= GH || c < 0 || c >= GW || grid[r][c].hp <= 0)
        return 0;
    return phase != BUILD || phase_t >= grid[r][c].delay + 0.3;
}

/* First solid brick overlapping a ball at (x, y), or -1 */
static int brick_at(double x, double y)
{
    int c0 = (int)floor((x - BALL_R - GRID_X0) / CW), c1 = (int)floor((x + BALL_R - 0.01 - GRID_X0) / CW);
    int r0 = (int)floor((y - BALL_R - GRID_Y0) / CH), r1 = (int)floor((y + BALL_R - 0.01 - GRID_Y0) / CH);
    for (int r = r0; r <= r1; r++)
        for (int c = c0; c <= c1; c++)
            if (brick_solid(r, c))
                return r * GW + c;
    return -1;
}

/* ------------------------------------------------ caches */

static void draw_brick(cairo_t *cr, double x, double y, rgb col, int cracked)
{
    rgb hi = lerp(col, WHITE, 0.5), lo = lerp(col, (rgb){ 0, 0, 0 }, 0.45);
    if (cracked)
        col = lerp(col, (rgb){ 0.2, 0.2, 0.25 }, 0.35);
    set_rgb(cr, lo);
    cairo_rectangle(cr, x + 1, y + 1, CW - 2, CH - 2);
    cairo_fill(cr);
    set_rgb(cr, hi);
    cairo_rectangle(cr, x + 1, y + 1, CW - 4, CH - 4);
    cairo_fill(cr);
    set_rgb(cr, col);
    cairo_rectangle(cr, x + 3, y + 3, CW - 6, CH - 6);
    cairo_fill(cr);
    if (cracked) {
        cairo_set_source_rgb(cr, 0.15, 0.15, 0.2);
        cairo_rectangle(cr, x + 7, y + 3, 2, 4);
        cairo_rectangle(cr, x + 9, y + 6, 2, 3);
        cairo_rectangle(cr, x + 11, y + 8, 2, 3);
        cairo_fill(cr);
    }
}

static void render_bricks(cairo_t *cr, int building)
{
    for (int r = 0; r < GH; r++)
        for (int c = 0; c < GW; c++) {
            const brick *b = &grid[r][c];
            if (b->hp <= 0)
                continue;
            double x = GRID_X0 + c * CW, y = GRID_Y0 + r * CH;
            if (building) {
                double u = (phase_t - b->delay) / 0.3;
                if (u <= 0)
                    continue;
                if (u < 1) {
                    /* drop in from above, snapped to 2 px steps so it reads as pixel art */
                    y -= floor((1 - u) * (1 - u) * 60 / 2) * 2;
                }
            }
            draw_brick(cr, x, y, b->col, b->silver && b->hp == 1);
        }
}

static cairo_surface_t *make_glow(rgb c)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 48, 48);
    cairo_t *cr = cairo_create(s);
    cairo_pattern_t *g = cairo_pattern_create_radial(24, 24, 2, 24, 24, 24);
    cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, 0.55);
    cairo_pattern_add_color_stop_rgba(g, 0.4, c.r, c.g, c.b, 0.2);
    cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

static void build_caches(void)
{
    srand(7);

    /* Background: deep space, a faint dot grid, stars, and the paddle track */
    bg_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(bg_cache);
    cairo_pattern_t *g = cairo_pattern_create_radial(CX, CY, 20, CX, CY, 240);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.06, 0.05, 0.14);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.01, 0.01, 0.04);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_set_source_rgba(cr, 0.35, 0.40, 0.9, 0.16);
    for (int y = 12; y < SIZE; y += 24)
        for (int x = 12; x < SIZE; x += 24)
            cairo_rectangle(cr, x - 1, y - 1, 2, 2);
    cairo_fill(cr);
    for (int i = 0; i < 70; i++) {
        double x = floor(frand() * SIZE / 2) * 2, y = floor(frand() * SIZE / 2) * 2, b = 0.25 + frand() * 0.5;
        cairo_set_source_rgb(cr, b, b, b * 1.1);
        cairo_rectangle(cr, x, y, frand() < 0.15 ? 3 : 2, 2);
        cairo_fill(cr);
    }
    /* paddle track: dim ring with tick marks, and the dividing line at top and bottom */
    cairo_set_line_width(cr, PAD_T + 4);
    cairo_set_source_rgb(cr, 0.07, 0.07, 0.13);
    cairo_arc(cr, CX, CY, R_PAD + PAD_T / 2, 0, 2 * M_PI);
    cairo_stroke(cr);
    cairo_set_line_width(cr, 2);
    for (int i = 0; i < 72; i++) {
        double a = i * 2 * M_PI / 72;
        cairo_move_to(cr, CX + cos(a) * (R_PAD + 3), CY + sin(a) * (R_PAD + 3));
        cairo_line_to(cr, CX + cos(a) * (R_PAD + PAD_T - 3), CY + sin(a) * (R_PAD + PAD_T - 3));
    }
    cairo_set_source_rgb(cr, 0.13, 0.13, 0.22);
    cairo_stroke(cr);
    cairo_set_source_rgba(cr, 0.5, 0.5, 0.7, 0.18);
    for (double y = 6; y < 150; y += 12) {
        cairo_rectangle(cr, CX - 1, y, 2, 6);
        cairo_rectangle(cr, CX - 1, SIZE - y - 6, 2, 6);
    }
    cairo_fill(cr);
    cairo_destroy(cr);

    /* Overlay: CRT scanlines, the cabinet bezel ring, black outside the circle */
    overlay_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cr = cairo_create(overlay_cache);
    cairo_set_source_rgba(cr, 0, 0, 0, 0.22);
    for (int y = 0; y < SIZE; y += 3)
        cairo_rectangle(cr, 0, y, SIZE, 1);
    cairo_fill(cr);
    cairo_set_fill_rule(cr, CAIRO_FILL_RULE_EVEN_ODD);
    cairo_rectangle(cr, 0, 0, SIZE, SIZE);
    cairo_arc(cr, CX, CY, 224, 0, 2 * M_PI);
    g = cairo_pattern_create_radial(CX, CY, 224, CX, CY, 240);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.30, 0.30, 0.38);
    cairo_pattern_add_color_stop_rgb(g, 0.25, 0.12, 0.12, 0.18);
    cairo_pattern_add_color_stop_rgb(g, 0.8, 0.06, 0.06, 0.09);
    cairo_pattern_add_color_stop_rgb(g, 1, 0, 0, 0);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);

    brick_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    text_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    for (int i = 0; i < 3; i++)
        glow[i] = make_glow(OWNER_COL[i]);

    srand((unsigned)time(NULL));
}

/* ---------------------------------------------------------------- simulation */

static void spawn_part(double x, double y, double vx, double vy, double life, double size, rgb col, int flash)
{
    for (int i = 0; i < MAX_PARTS; i++)
        if (parts[i].life <= 0) {
            parts[i] = (particle){ x, y, vx, vy, life, life, size, col, flash };
            return;
        }
}

static void spawn_ball(int owner, double a, int bonus, double speed)
{
    for (int i = 0; i < MAX_BALLS; i++) {
        ball *b = &balls[i];
        if (b->alive)
            continue;
        double dir = a + M_PI + (frand() - 0.5) * 0.9;
        memset(b, 0, sizeof(*b));
        b->x = CX + cos(a) * (R_HIT - 2);
        b->y = CY + sin(a) * (R_HIT - 2);
        b->speed = speed;
        b->vx = cos(dir) * speed;
        b->vy = sin(dir) * speed;
        for (int k = 0; k < 3; k++)
            b->tx[k] = b->x, b->ty[k] = b->y;
        b->owner = owner;
        b->bonus = bonus;
        b->aim = (frand() - 0.5) * 1.2;
        b->alive = 1;
        return;
    }
}

static void break_brick(int idx, int owner)
{
    int r = idx / GW, c = idx % GW;
    brick *b = &grid[r][c];
    double x = GRID_X0 + (c + 0.5) * CW, y = GRID_Y0 + (r + 0.5) * CH;
    spawn_part(x, y, 0, 0, 0.15, 0, WHITE, 1);
    bricks_dirty = 1;
    if (--b->hp > 0)
        return;
    for (int k = 0; k < 7; k++) {
        double a = frand() * 2 * M_PI, v = 50 + frand() * 140;
        spawn_part(x + (frand() - 0.5) * 10, y + (frand() - 0.5) * 8, cos(a) * v, sin(a) * v,
                   0.5 + frand() * 0.5, 4 + (rand() % 2) * 2, k < 2 ? OWNER_COL[owner] : b->col, 0);
    }
    bricks_left--;
    if (frand() < 0.08)
        for (int i = 0; i < MAX_CAPS; i++)
            if (!caps[i].alive) {
                double a = atan2(y - CY, x - CX) + (frand() - 0.5) * 0.6;
                caps[i] = (capsule){ x, y, cos(a) * 62, sin(a) * 62, rand() % 2, 1 };
                break;
            }
}

/* Where (angle) and when a thing at (x, y) moving (vx, vy) reaches the paddle radius */
static int impact(double x, double y, double vx, double vy, double rad, double *ang, double *t)
{
    double px = x - CX, py = y - CY;
    double a = vx * vx + vy * vy, b = 2 * (px * vx + py * vy), c = px * px + py * py - rad * rad;
    double d = b * b - 4 * a * c;
    if (a < 1e-6 || d < 0)
        return 0;
    *t = (-b + sqrt(d)) / (2 * a);
    if (*t < 0)
        return 0;
    *ang = atan2(py + vy * *t, px + vx * *t);
    return 1;
}

/* The paddle whose half of the ring contains angle a: 0 = left, 1 = right */
static int half_of(double a) { return cos(a) < 0 ? 0 : 1; }

static double pad_home(int p) { return p == 0 ? M_PI : 0; }

static void pad_limits(int p, double *lo, double *hi)
{
    double hw = pads[p].hw;
    *lo = (p == 0 ? M_PI / 2 : -M_PI / 2) + hw;
    *hi = (p == 0 ? 3 * M_PI / 2 : M_PI / 2) - hw;
}

/* Bring an angle into paddle p's continuous range */
static double pad_angle(int p, double a)
{
    double h = pad_home(p);
    return h + wrap_pi(a - h);
}

static int paddle_covers(int p, double a, double margin)
{
    return fabs(wrap_pi(a - pads[p].a)) <= pads[p].hw + margin;
}

static void simulate(const stats *s, double dt)
{
    int idle = s->tok_s < 1 && s->running == 0;
    int target[3], have[3] = { 0, 0, 0 };
    double speed[3];

    for (int p = 0; p < 2; p++) {
        double tk = s->tok_port[p];
        target[p] = tk >= 1 ? 1 + (int)(tk / TOK_PER_BALL) : (s->running_port[p] > 0 ? 1 : 0);
        if (target[p] > MAX_OWN_BALLS)
            target[p] = MAX_OWN_BALLS;
        speed[p] = fmin(300, 160 + tk * 0.18);
    }
    target[2] = idle ? 1 : 0;
    speed[2] = 125;
    score += s->tok_s * dt;

    /* stage flow */
    phase_t += dt;
    if (phase == PLAY && bricks_left <= 0) {
        phase = CLEAR;
        phase_t = 0;
    } else if (phase == CLEAR && phase_t > CLEAR_TIME) {
        load_stage(++stage);
    } else if (phase == BUILD) {
        bricks_dirty = 1;
        if (phase_t > BUILD_TIME) {
            phase = PLAY;
            phase_t = 0;
        }
    }

    /* serve new balls from the paddles */
    for (int i = 0; i < MAX_BALLS; i++)
        if (balls[i].alive && !balls[i].lost && !balls[i].bonus)
            have[balls[i].owner]++;
    for (int o = 0; o < 3; o++) {
        serve_cd[o] -= dt;
        if (have[o] < target[o] && serve_cd[o] <= 0) {
            int p = o < 2 ? o : (serve_alt ^= 1);
            spawn_ball(o, pads[p].a, 0, speed[o]);
            pads[p].flash = 1;
            serve_cd[o] = 0.4;
            have[o]++;
        }
    }

    /* paddle AI: chase the ball (or capsule) that reaches its half of the rim first,
     * skipping balls it can no longer get to in time */
    for (int p = 0; p < 2; p++) {
        paddle *pd = &pads[p];
        double best = 1e9, best_any = 1e9, want = pad_home(p), want_any = want, lo, hi;
        double vmax = 2.4 + 1.6 * s->load[p];
        pd->wide_t -= dt;
        double hw_want = pd->wide_t > 0 ? PAD_HW * 1.5 : PAD_HW;
        pd->hw += (hw_want - pd->hw) * fmin(1, dt * 6);
        pd->flash = fmax(0, pd->flash - dt * 4);
        for (int i = 0; i < MAX_BALLS + MAX_CAPS; i++) {
            double a, t, x, y, vx, vy, rad = R_HIT, aim = 0;
            if (i < MAX_BALLS) {
                const ball *b = &balls[i];
                if (!b->alive || b->lost)
                    continue;
                x = b->x, y = b->y, vx = b->vx, vy = b->vy, aim = b->aim;
            } else {
                const capsule *cp = &caps[i - MAX_BALLS];
                if (!cp->alive)
                    continue;
                x = cp->x, y = cp->y, vx = cp->vx, vy = cp->vy, rad = R_PAD - 6;
            }
            if (!impact(x, y, vx, vy, rad, &a, &t) || half_of(a) != p)
                continue;
            if (t < best_any)
                best_any = t, want_any = a;
            double gap = fabs(pad_angle(p, a) - pd->a) - pd->hw * 0.8;
            if (t < best && gap <= t * vmax * 0.9 + 0.02)
                best = t, want = a + aim * pd->hw;
        }
        if (best > 1e8)
            want = want_any;
        pad_limits(p, &lo, &hi);
        want = pad_angle(p, want);
        want = want < lo ? lo : want > hi ? hi : want;
        double vwant = (want - pd->a) * 9;
        vwant = vwant > vmax ? vmax : vwant < -vmax ? -vmax : vwant;
        double acc = 30 * dt;
        pd->va += (vwant - pd->va) > acc ? acc : (vwant - pd->va) < -acc ? -acc : (vwant - pd->va);
        pd->a += pd->va * dt;
        if (pd->a < lo) pd->a = lo, pd->va = 0;
        if (pd->a > hi) pd->a = hi, pd->va = 0;
    }

    /* balls */
    for (int i = 0; i < MAX_BALLS; i++) {
        ball *b = &balls[i];
        if (!b->alive)
            continue;
        b->tx[2] = b->tx[1]; b->ty[2] = b->ty[1];
        b->tx[1] = b->tx[0]; b->ty[1] = b->ty[0];
        b->tx[0] = b->x;     b->ty[0] = b->y;
        int steps = (int)ceil(b->speed * dt / 3.0);
        double h = dt / (steps > 0 ? steps : 1);
        for (int k = 0; k < steps && b->alive; k++) {
            /* bricks: move one axis at a time and bounce off whichever axis hit */
            int inside = brick_at(b->x, b->y) >= 0, idx;
            b->x += b->vx * h;
            if (!inside && (idx = brick_at(b->x, b->y)) >= 0) {
                b->x -= b->vx * h;
                b->vx = -b->vx;
                break_brick(idx, b->owner);
            }
            b->y += b->vy * h;
            if (!inside && (idx = brick_at(b->x, b->y)) >= 0) {
                b->y -= b->vy * h;
                b->vy = -b->vy;
                break_brick(idx, b->owner);
            }
            /* paddles */
            double px = b->x - CX, py = b->y - CY, r = sqrt(px * px + py * py);
            if (!b->lost && r >= R_HIT && px * b->vx + py * b->vy > 0) {
                double a = atan2(py, px);
                int p = -1;
                for (int q = 0; q < 2; q++)
                    if (paddle_covers(q, a, BALL_R / R_PAD * 1.3))
                        p = q;
                if (p < 0) {
                    b->lost = 1;                        /* missed: it sails off the screen */
                    misses++;
                    continue;
                }
                pads[p].flash = 1;
                int o = b->owner;
                if ((o < 2 && target[o] == 0) || (o == 2 && !idle) ||
                    (!b->bonus && have[o] > target[o])) {
                    /* not needed any more: the paddle catches it */
                    have[o] -= !b->bonus;
                    b->alive = 0;
                    for (int n = 0; n < 6; n++)
                        spawn_part(b->x, b->y, (frand() - 0.5) * 120, (frand() - 0.5) * 120, 0.35, 4, OWNER_COL[o], 0);
                    continue;
                }
                double off = wrap_pi(a - pads[p].a) / pads[p].hw;
                off = off < -1 ? -1 : off > 1 ? 1 : off;
                double dir = a + M_PI + off * 0.85 + (frand() - 0.5) * 0.12;
                /* the AI aims for the leftovers, more keenly as the picture thins out */
                if (phase == PLAY && bricks_left > 0 && frand() < (bricks_left < 16 ? 0.75 : 0.3)) {
                    int pick = rand() % bricks_left;
                    for (int n = 0; n < GH * GW; n++)
                        if (grid[n / GW][n % GW].hp > 0 && pick-- == 0) {
                            dir = atan2(GRID_Y0 + (n / GW + 0.5) * CH - b->y, GRID_X0 + (n % GW + 0.5) * CW - b->x) +
                                  (frand() - 0.5) * 0.1;
                            break;
                        }
                }
                b->speed = b->bonus ? fmax(speed[o], 180) : speed[o];
                b->vx = cos(dir) * b->speed;
                b->vy = sin(dir) * b->speed;
                b->x = CX + cos(a) * (R_HIT - 0.5);
                b->y = CY + sin(a) * (R_HIT - 0.5);
                b->aim = (frand() - 0.5) * 1.2;
                for (int n = 0; n < 3; n++)
                    spawn_part(b->x, b->y, cos(dir + (frand() - 0.5) * 1.6) * 90, sin(dir + (frand() - 0.5) * 1.6) * 90,
                               0.25, 3, OWNER_COL[o], 0);
            }
            if (b->lost && r > 222) {
                /* burst against the bezel */
                for (int n = 0; n < 8; n++) {
                    double a = atan2(py, px) + M_PI + (frand() - 0.5) * 2.4, v = 60 + frand() * 120;
                    spawn_part(b->x, b->y, cos(a) * v, sin(a) * v, 0.45, 4, n & 1 ? WHITE : OWNER_COL[b->owner], 0);
                }
                b->alive = 0;
            }
        }
    }

    /* capsules drift outward; a paddle that catches one gets the power-up */
    for (int i = 0; i < MAX_CAPS; i++) {
        capsule *cp = &caps[i];
        if (!cp->alive)
            continue;
        cp->x += cp->vx * dt;
        cp->y += cp->vy * dt;
        double px = cp->x - CX, py = cp->y - CY, r = sqrt(px * px + py * py);
        if (r >= R_PAD - 6 && r < R_PAD + 2) {
            double a = atan2(py, px);
            for (int p = 0; p < 2; p++)
                if (paddle_covers(p, a, 0.04)) {
                    cp->alive = 0;
                    pads[p].flash = 1;
                    for (int n = 0; n < 10; n++)
                        spawn_part(cp->x, cp->y, (frand() - 0.5) * 200, (frand() - 0.5) * 200, 0.5, 4,
                                   cp->type ? (rgb){ 0.4, 1, 0.5 } : (rgb){ 1, 0.4, 0.9 }, 0);
                    if (cp->type == 1) {
                        pads[p].wide_t = 10;
                    } else {
                        /* multiball: every ball of this paddle's server (or the attract ball) splits */
                        int owner = idle ? 2 : p, n0 = 0, bonus = 0;
                        for (int k = 0; k < MAX_BALLS; k++)
                            bonus += balls[k].alive && balls[k].bonus;
                        for (int k = 0; k < MAX_BALLS && n0 < 3 && bonus + n0 < 6; k++) {
                            ball *b = &balls[k];
                            if (b->alive && !b->lost && b->owner == owner) {
                                spawn_ball(owner, atan2(b->y - CY, b->x - CX), 1, fmax(b->speed, 180));
                                n0++;
                            }
                        }
                        if (!n0 && bonus < 6)
                            spawn_ball(owner, pads[p].a, 1, 200);
                    }
                    break;
                }
        }
        if (r > 250)
            cp->alive = 0;
    }

    for (int i = 0; i < MAX_PARTS; i++) {
        particle *pt = &parts[i];
        if (pt->life <= 0)
            continue;
        pt->life -= dt;
        pt->x += pt->vx * dt;
        pt->y += pt->vy * dt;
        pt->vx *= 1 - dt * 1.5;
        pt->vy *= 1 - dt * 1.5;
    }

    bulb_phase += dt * (idle ? 1.5 : 3 + fmin(s->tok_s, 2000) / 110);
}

/* ---------------------------------------------------------------- render */

typedef struct { double zotac, tuf, tok; } shown_t;

static char text_key[256];

static void fmt_score(char *out, size_t n, double v)
{
    if (v >= 1e8)
        snprintf(out, n, "%.1fM", v / 1e6);
    else
        snprintf(out, n, "%08.0f", v);
}

/* Redraw the text layer only when what it says changes */
static void update_text(const stats *s, const shown_t *sh, double t)
{
    int idle = s->tok_s < 1 && s->running == 0;
    int blink = fmod(t, 1.2) < 0.8;
    char top[32], a[16], b[16], sc[24], key[256];

    snprintf(top, sizeof(top), "%.0f", shown_rate(sh->tok));
    snprintf(a, sizeof(a), "%.0f", shown_gpu_rate(sh->zotac));
    snprintf(b, sizeof(b), "%.0f", shown_gpu_rate(sh->tuf));
    fmt_score(sc, sizeof(sc), score);
    snprintf(key, sizeof(key), "%d|%s|%s|%s|%s|%d|%d|%d", idle, top, a, b, idle ? sc : "", blink, phase, stage);
    if (!strcmp(key, text_key))
        return;
    snprintf(text_key, sizeof(text_key), "%s", key);

    cairo_t *cr = cairo_create(text_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    const rgb YEL = { 1.0, 0.86, 0.25 }, RED = { 1.0, 0.30, 0.30 };
    if (idle) {
        ptext(cr, CX, 58, 2.6, RED, "HI-SCORE");
        ptext(cr, CX, 84, 3.6, WHITE, sc);
        if (blink)
            ptext(cr, CX, 364, 3.4, YEL, "INSERT COIN");
    } else {
        ptext(cr, CX, 50, 5.6, WHITE, top);
        ptext(cr, CX, 97, 2.4, YEL, rate_unit_uc());
        ptext(cr, CX - 64, 346, 2.4, OWNER_COL[0], "1UP");
        ptext(cr, CX + 64, 346, 2.4, OWNER_COL[1], "2UP");
        ptext(cr, CX - 64, 368, 4.2, OWNER_COL[0], a);
        ptext(cr, CX + 64, 368, 4.2, OWNER_COL[1], b);
    }
    if (phase == CLEAR) {
        ptext(cr, CX, 206, 5.2, YEL, "CLEAR!");
    } else if (phase == BUILD && stage > 1) {
        char st[24];
        snprintf(st, sizeof(st), "STAGE %d", stage);
        ptext(cr, CX, 200, 5, WHITE, st);
        ptext(cr, CX, 250, 3, YEL, "READY!");
    }
    cairo_destroy(cr);
}

static void render_paddle(cairo_t *cr, int p, double power)
{
    const paddle *pd = &pads[p];
    rgb col = OWNER_COL[p];
    double rm = R_PAD + PAD_T / 2, a0 = pd->a - pd->hw, a1 = pd->a + pd->hw, cap = 0.075;

    /* glow under the paddle, stronger with GPU power */
    cairo_set_line_width(cr, PAD_T + 10);
    cairo_set_source_rgba(cr, col.r, col.g, col.b, 0.15 + 0.25 * clamp01(power / 450) + 0.3 * pd->flash);
    cairo_arc(cr, CX, CY, rm, a0 - 0.02, a1 + 0.02);
    cairo_stroke(cr);

    /* body, with silver-and-red end caps like the classic bat */
    cairo_set_line_width(cr, PAD_T);
    set_rgb(cr, lerp(col, WHITE, 0.45 * pd->flash));
    cairo_arc(cr, CX, CY, rm, a0 + cap, a1 - cap);
    cairo_stroke(cr);
    cairo_set_source_rgb(cr, 0.92, 0.25, 0.25);
    cairo_arc(cr, CX, CY, rm, a0, a0 + cap);
    cairo_stroke(cr);
    cairo_arc(cr, CX, CY, rm, a1 - cap, a1);
    cairo_stroke(cr);
    cairo_set_source_rgb(cr, 0.80, 0.82, 0.88);
    cairo_set_line_width(cr, PAD_T);
    cairo_arc(cr, CX, CY, rm, a0 + cap - 0.018, a0 + cap + 0.012);
    cairo_stroke(cr);
    cairo_arc(cr, CX, CY, rm, a1 - cap - 0.012, a1 - cap + 0.018);
    cairo_stroke(cr);
    /* inner highlight and outer shade */
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.55);
    cairo_arc(cr, CX, CY, R_PAD + 2, a0 + 0.01, a1 - 0.01);
    cairo_stroke(cr);
    cairo_set_source_rgba(cr, 0, 0, 0, 0.35);
    cairo_arc(cr, CX, CY, R_PAD + PAD_T - 2, a0 + 0.01, a1 - 0.01);
    cairo_stroke(cr);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    cairo_set_source_surface(cr, bg_cache, 0, 0);
    cairo_paint(cr);

    /* bricks */
    if (phase == BUILD) {
        render_bricks(cr, 1);
    } else {
        if (bricks_dirty) {
            cairo_t *bc = cairo_create(brick_cache);
            cairo_set_operator(bc, CAIRO_OPERATOR_CLEAR);
            cairo_paint(bc);
            cairo_set_operator(bc, CAIRO_OPERATOR_OVER);
            render_bricks(bc, 0);
            cairo_destroy(bc);
            bricks_dirty = 0;
        }
        cairo_set_source_surface(cr, brick_cache, 0, 0);
        cairo_paint(cr);
    }

    /* particles */
    for (int i = 0; i < MAX_PARTS; i++) {
        const particle *pt = &parts[i];
        if (pt->life <= 0)
            continue;
        double k = pt->life / pt->max;
        if (pt->flash) {
            cairo_set_source_rgba(cr, 1, 1, 1, 0.9 * k);
            cairo_rectangle(cr, pt->x - CW / 2 - 1, pt->y - CH / 2 - 1, CW + 2, CH + 2);
        } else {
            cairo_set_source_rgba(cr, pt->col.r, pt->col.g, pt->col.b, fmin(1, k * 1.6));
            cairo_rectangle(cr, floor(pt->x - pt->size / 2), floor(pt->y - pt->size / 2), pt->size, pt->size);
        }
        cairo_fill(cr);
    }

    /* capsules */
    for (int i = 0; i < MAX_CAPS; i++) {
        const capsule *cp = &caps[i];
        if (!cp->alive)
            continue;
        rgb c = cp->type ? (rgb){ 0.30, 0.90, 0.40 } : (rgb){ 0.95, 0.35, 0.85 };
        double x = floor(cp->x - 15), y = floor(cp->y - 9);
        cairo_new_path(cr);
        cairo_arc(cr, x + 9, y + 9, 9, M_PI / 2, 3 * M_PI / 2);
        cairo_arc(cr, x + 21, y + 9, 9, -M_PI / 2, M_PI / 2);
        cairo_close_path(cr);
        cairo_set_source_rgb(cr, 0, 0, 0);
        cairo_set_line_width(cr, 3);
        cairo_stroke_preserve(cr);
        set_rgb(cr, c);
        cairo_fill(cr);
        cairo_set_source_rgba(cr, 1, 1, 1, 0.45);
        cairo_rectangle(cr, x + 6, y + 3, 18, 3);
        cairo_fill(cr);
        ptext(cr, x + 15, y + 4.5, 1.4, WHITE, cp->type ? "W" : "M");
    }

    /* balls: glow, trail, square pixel ball */
    for (int i = 0; i < MAX_BALLS; i++) {
        const ball *b = &balls[i];
        if (!b->alive)
            continue;
        rgb c = OWNER_COL[b->owner];
        cairo_set_source_surface(cr, glow[b->owner], floor(b->x) - 24, floor(b->y) - 24);
        cairo_paint(cr);
        for (int k = 2; k >= 0; k--) {
            double sz = BALL_R * 2 * (0.75 - k * 0.15);
            cairo_set_source_rgba(cr, c.r, c.g, c.b, 0.45 - k * 0.12);
            cairo_rectangle(cr, floor(b->tx[k] - sz / 2), floor(b->ty[k] - sz / 2), sz, sz);
            cairo_fill(cr);
        }
        double x = floor(b->x - BALL_R), y = floor(b->y - BALL_R);
        cairo_set_source_rgb(cr, 0, 0, 0);
        cairo_rectangle(cr, x - 1, y - 1, BALL_R * 2 + 2, BALL_R * 2 + 2);
        cairo_fill(cr);
        set_rgb(cr, lerp(c, WHITE, 0.35));
        cairo_rectangle(cr, x, y, BALL_R * 2, BALL_R * 2);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_rectangle(cr, x + 3, y + 3, 4, 4);
        cairo_fill(cr);
    }

    for (int p = 0; p < 2; p++)
        render_paddle(cr, p, s->power[p]);

    update_text(s, sh, t);
    cairo_set_source_surface(cr, text_cache, 0, 0);
    cairo_paint(cr);

    cairo_set_source_surface(cr, overlay_cache, 0, 0);
    cairo_paint(cr);

    /* marquee bulbs chasing around the bezel: blue half, orange half, white when idle */
    int idle = s->tok_s < 1 && s->running == 0;
    for (int i = 0; i < N_BULBS; i++) {
        double a = (i + 0.5) * 2 * M_PI / N_BULBS - M_PI / 2;
        double lit = 0.5 + 0.5 * cos((i - bulb_phase) * 2 * M_PI / 6.0);
        lit = lit * lit;
        rgb c = idle ? (rgb){ 1, 0.85, 0.5 } : OWNER_COL[half_of(a)];
        double x = CX + cos(a) * 231, y = CY + sin(a) * 231;
        cairo_set_source_rgb(cr, c.r * (0.25 + 0.75 * lit), c.g * (0.25 + 0.75 * lit), c.b * (0.25 + 0.75 * lit));
        cairo_arc(cr, x, y, 3.2, 0, 2 * M_PI);
        cairo_fill(cr);
    }
}

/* ---------------------------------------------------------------- showcase */

/* A scripted 40 s loop through every state, for filming: tok/s per server at each time */
typedef struct { double t, tok0, tok1; } keyframe;

static const keyframe script[] = {
    {  0.0,    0,   0 },            /* attract mode */
    {  5.0,    0,   0 },
    {  5.5,  150,   0 },            /* GPU 0 wakes up */
    {  9.0,  260,  90 },
    { 13.0,  700, 600 },            /* both flat out */
    { 30.0,  820, 900 },
    { 34.0,   60,  40 },
    { 35.0,    0,   0 },            /* back to attract mode */
    { 40.0,    0,   0 },
};
#define SCRIPT_LEN      (sizeof(script) / sizeof(script[0]))
#define SCRIPT_PERIOD   40.0

static void showcase_poll(stats *s, double t)
{
    double lt = fmod(t, SCRIPT_PERIOD), t0 = 0, t1 = 0;
    for (size_t i = 0; i + 1 < SCRIPT_LEN; i++) {
        const keyframe *a = &script[i], *b = &script[i + 1];
        if (lt >= a->t && lt < b->t) {
            double u = (lt - a->t) / (b->t - a->t);
            u = u * u * (3 - 2 * u);
            t0 = a->tok0 + (b->tok0 - a->tok0) * u;
            t1 = a->tok1 + (b->tok1 - a->tok1) * u;
            break;
        }
    }
    double wobble = 1 + 0.05 * sin(t * 1.7);
    s->tok_port[0] = t0 < 1 ? 0 : t0 * wobble;
    s->tok_port[1] = t1 < 1 ? 0 : t1 / wobble;
    s->tok_s = s->tok_port[0] + s->tok_port[1];
    s->running = s->tok_s > 0 ? 1 + (int)(s->tok_s / 250) : 0;
    s->running_port[0] = s->running_port[1] = 0;
    for (int i = 0; i < 2; i++) {
        double tk = s->tok_port[i];
        s->load[i] = clamp01(tk / 600);
        s->power[i] = 30 + fmin(540, tk * 0.6);
        s->temp[i] = 38 + (int)(tk / 25);
    }
}

/* ---------------------------------------------------------------- main */

static void reset_game(int stage_no)
{
    memset(balls, 0, sizeof(balls));
    memset(parts, 0, sizeof(parts));
    memset(caps, 0, sizeof(caps));
    pads[0] = (paddle){ M_PI, 0, PAD_HW, 0, 0 };
    pads[1] = (paddle){ 0, 0, PAD_HW, 0, 0 };
    stage = stage_no;
    load_stage(stage);
    phase = PLAY;
    serve_cd[0] = serve_cd[1] = serve_cd[2] = 0;
    text_key[0] = 0;
}

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats s = { 0 };
    shown_t sh = { 0 };
    double last, next_poll = 0, next_text = 0, t0;
    int fd = -1, bench = 0, showcase = 0;

    const char *source = getenv("LLM_REACTOR_SOURCE");
    gpu_source = source && !strcasecmp(source, "gpu");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu-load"))
            gpu_source = 1;
        else if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
        else if (!strcmp(argv[i], "--showcase"))
            showcase = demo = 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    build_caches();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok0, tok1, w0, w1; int stage; double secs; const char *png; } scenes[] = {
            { 780, 860, 470, 460, 1, 3.5, "brickout_heavy.png" },
            { 240, 90, 250, 120, 2, 5, "brickout_light.png" },
            { 0, 0, 25, 30, 4, 6.5, "brickout_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(42 + k);
            reset_game(scenes[k].stage);
            score = 1234567 + k * 777;
            memset(&s, 0, sizeof(s));
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = s.tok_s > 0 ? 4 : 0;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1; s.temp[0] = 54; s.temp[1] = 71;
            s.load[0] = clamp01(scenes[k].tok0 / 600); s.load[1] = clamp01(scenes[k].tok1 / 600);
            sh.tok = s.tok_s; sh.zotac = s.tok_port[0]; sh.tuf = s.tok_port[1];
            int n = (int)(scenes[k].secs * FPS_BUSY), nb = 0;
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                if (i == n - 60)
                    b0 = now_s();
                simulate(&s, 1.0 / FPS_BUSY);
                render(cr, &s, &sh, i / (double)FPS_BUSY);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            for (int i = 0; i < MAX_BALLS; i++)
                nb += balls[i].alive;
            printf("%s: %.2f ms/frame, jpeg %zu bytes, %d balls, %d bricks left, %d misses\n", scenes[k].png,
                   (now_s() - b0) * 1000 / 60, len, nb, bricks_left, misses);
            misses = 0;
            cairo_surface_write_to_png(surf, scenes[k].png);
        }
        return 0;
    }

    if (!demo)
        gpus_init();

    while ((fd = lcd_open()) < 0 && !stop)
        sleep(2);
    if (fd >= 0)
        lcd_brightness(fd, 100);

    reset_game(1);
    phase = BUILD;
    t0 = last = now_s();
    while (!stop) {
        double t = now_s(), dt = t - last;
        last = t;
        if (dt > 0.25)
            dt = 0.25;

        if (showcase) {
            showcase_poll(&s, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 1.0;
            if (demo) {
                /* scale a copy each second; scaling s itself would compound */
                static stats demo_base;
                demo_poll(&demo_base, t - t0);
                s = demo_base;
                s.tok_port[0] *= 6; s.tok_port[1] *= 6;
                s.tok_s = s.tok_port[0] + s.tok_port[1];
                if (s.tok_s < 150) { s.tok_s = s.tok_port[0] = s.tok_port[1] = 0; s.running = 0; }
            } else {
                gpus_poll(&s);
                if (gpu_source)
                    gpu_rate_poll(&s);
                else
                    vllm_poll(&s, t);
            }
        }

        /* the numbers on screen ease toward the data and change 4x a second */
        if (t >= next_text) {
            next_text = t + 0.25;
            double k = showcase ? 1 : 0.6;
            sh.tok   += (s.tok_s - sh.tok) * k;
            sh.zotac += (s.tok_port[0] - sh.zotac) * k;
            sh.tuf   += (s.tok_port[1] - sh.tuf) * k;
        }

        simulate(&s, dt);
        render(cr, &s, &sh, t - t0);
        cairo_surface_flush(surf);
        size_t len = 0;
        if (tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                         SIZE, TJPF_BGRX, &jpeg, &len) == 0) {
            if (lcd_send(fd, jpeg, len) < 0) {
                close(fd);
                while ((fd = lcd_open()) < 0 && !stop)
                    sleep(2);
                if (fd >= 0)
                    lcd_brightness(fd, 100);
            }
        }

        /* Attract mode only has one slow ball, so it runs at a lower frame rate */
        int idle = !showcase && s.tok_s < 1 && s.running == 0 && phase == PLAY;
        double spare = 1.0 / (idle ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    cairo_surface_destroy(bg_cache);
    cairo_surface_destroy(overlay_cache);
    cairo_surface_destroy(brick_cache);
    cairo_surface_destroy(text_cache);
    for (int i = 0; i < 3; i++)
        cairo_surface_destroy(glow[i]);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
