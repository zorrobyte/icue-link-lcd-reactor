/*
 * knit: grandma knits an impossibly long scarf on the iCUE LINK AIO pump LCD.
 *
 * Every generated token is a stitch. Her needles click faster with tok/s (or GPU
 * activity in GPU mode) and the scarf pours off her lap and coils across the rug and
 * out of frame. Its stripes are knitted from two balls of yarn, one per GPU: each
 * ball is as big as that GPU's free VRAM, rolls as it unwinds and glows warmer with
 * its temperature. The rest of the machine keeps house: the fire burns with CPU
 * temperature, the teapot steams with CPU load, the basket fills with yarn as RAM
 * fills, the radio plays network traffic, the cat goes wild with NVMe activity,
 * context switches make her change yarn more often (narrower stripes), bursts of
 * major page faults make her drop a stitch (a ladder runs down the scarf) and the
 * wall calendar counts the days of uptime. The sampler on the wall stitches the
 * rate and the total. Idle, she dozes off and the cat curls up on the scarf.
 *
 * Art (room, grandma, cat, props) was generated with an image model (assets/knit/).
 * Run with --demo to simulate data, --showcase for a scripted 36 s arc, --bench to
 * write preview PNGs.
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

/* ---- system sensors ---- */

/*
 * Everything that isn't vLLM or GPU load comes from the kernel (plus NVML's VRAM):
 * per-thread CPU load, context switches (/proc/stat), CPU temperature (hwmon named
 * k10temp, found by name), RAM (/proc/meminfo), NVMe throughput (/proc/diskstats),
 * network (/proc/net/dev), major page faults (/proc/vmstat) and uptime. sys_poll()
 * is cheap and meant for ~2 Hz; counters are differenced into rates. Missing sensors
 * stay at 0.
 */
#define SYS_MAX_THREADS 64
#define N_NVME          3

typedef struct {
    int    n_threads;
    double thread_load[SYS_MAX_THREADS];    /* 0..1 per hardware thread */
    double cpu_load;                        /* 0..1 whole package */
    int    busy_threads;                    /* threads above 50% */
    double cpu_temp;                        /* k10temp Tctl, C */
    double ram_total, ram_used;             /* GB (used = total - available) */
    double disk;                            /* nvme0n1..nvme2n1 read + write, bytes/s */
    double net_rx, net_tx;                  /* enp12s0, bytes/s */
    double ts_rx, ts_tx;                    /* tailscale0, bytes/s */
    double ctxt;                            /* context switches/s */
    double majflt;                          /* major page faults/s */
    double uptime;                          /* s */
    double vram[N_GPUS];                    /* NVML: fraction of each GPU's memory in use */
} sys_stats;

static const char *NET_IF = "enp12s0", *TS_IF = "tailscale0";
static char k10_temp_path[300];

static int read_text(const char *path, char *buf, int cap)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    ssize_t n = read(fd, buf, cap - 1);
    close(fd);
    if (n < 0)
        return -1;
    buf[n] = 0;
    return (int)n;
}

static void sys_init(void)
{
    char path[300], buf[128];
    DIR *d = opendir("/sys/class/hwmon");
    struct dirent *e;

    while (d && (e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5))
            continue;
        snprintf(path, sizeof(path), "/sys/class/hwmon/%s/name", e->d_name);
        if (read_text(path, buf, sizeof(buf)) <= 0)
            continue;
        buf[strcspn(buf, "\n")] = 0;
        if (!strcmp(buf, "k10temp"))
            snprintf(k10_temp_path, sizeof(k10_temp_path), "/sys/class/hwmon/%s/temp1_input", e->d_name);
    }
    if (d)
        closedir(d);
}

static void sys_poll(sys_stats *s, double t)
{
    static unsigned long long last_busy[SYS_MAX_THREADS], last_total[SYS_MAX_THREADS];
    static unsigned long long last_disk, last_net[4], last_ctxt, last_majflt;
    static double last_t;
    static int have;
    double dt = t - last_t;
    char line[512], buf[64];
    FILE *f;

    /* CPU: per-thread busy share since the last poll, and context switches */
    if ((f = fopen("/proc/stat", "r"))) {
        unsigned long long all_busy = 0, all_total = 0;
        int n = 0, busy_n = 0;
        while (fgets(line, sizeof(line), f)) {
            if (!strncmp(line, "ctxt ", 5)) {
                unsigned long long c = strtoull(line + 5, NULL, 10);
                if (have && dt > 0 && c >= last_ctxt)
                    s->ctxt = (c - last_ctxt) / dt;
                last_ctxt = c;
                break;                              /* after the cpu and intr lines */
            }
            if (strncmp(line, "cpu", 3) || line[3] == ' ')
                continue;
            unsigned long long v[8] = { 0 };
            int id;
            if (sscanf(line + 3, "%d %llu %llu %llu %llu %llu %llu %llu %llu", &id, &v[0], &v[1], &v[2], &v[3],
                       &v[4], &v[5], &v[6], &v[7]) < 9 || id < 0 || id >= SYS_MAX_THREADS)
                continue;
            unsigned long long total = 0, idle = v[3] + v[4];
            for (int k = 0; k < 8; k++)
                total += v[k];
            unsigned long long busy = total - idle;
            if (have && total > last_total[id]) {
                s->thread_load[id] = clamp01((double)(busy - last_busy[id]) / (double)(total - last_total[id]));
                all_busy += busy - last_busy[id];
                all_total += total - last_total[id];
            }
            last_busy[id] = busy;
            last_total[id] = total;
            busy_n += s->thread_load[id] > 0.5;
            if (id + 1 > n)
                n = id + 1;
        }
        fclose(f);
        s->n_threads = n;
        s->busy_threads = busy_n;
        if (all_total)
            s->cpu_load = (double)all_busy / all_total;
    }

    if (*k10_temp_path && read_text(k10_temp_path, buf, sizeof(buf)) > 0)
        s->cpu_temp = strtod(buf, NULL) / 1000.0;

    if (read_text("/proc/uptime", buf, sizeof(buf)) > 0)
        s->uptime = strtod(buf, NULL);

    /* RAM */
    if ((f = fopen("/proc/meminfo", "r"))) {
        double total = 0, avail = 0, v;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %lf", &v) == 1)
                total = v;
            else if (sscanf(line, "MemAvailable: %lf", &v) == 1) {
                avail = v;
                break;
            }
        }
        fclose(f);
        s->ram_total = total / 1048576.0;
        s->ram_used = (total - avail) / 1048576.0;
    }

    /* NVMe throughput: sectors read (field 6) and written (field 10), 512 bytes each */
    if ((f = fopen("/proc/diskstats", "r"))) {
        unsigned long long sum = 0;
        while (fgets(line, sizeof(line), f)) {
            char name[32];
            unsigned long long rd, wr;
            if (sscanf(line, "%*u %*u %31s %*u %*u %llu %*u %*u %*u %llu", name, &rd, &wr) != 3)
                continue;
            if (strncmp(name, "nvme", 4) || strlen(name) != 7 || strcmp(name + 5, "n1") ||
                name[4] < '0' || name[4] >= '0' + N_NVME)
                continue;
            sum += rd + wr;
        }
        fclose(f);
        if (have && dt > 0 && sum >= last_disk)
            s->disk = (sum - last_disk) * 512.0 / dt;
        last_disk = sum;
    }

    /* Network */
    if ((f = fopen("/proc/net/dev", "r"))) {
        while (fgets(line, sizeof(line), f)) {
            char *colon = strchr(line, ':'), *name = line;
            unsigned long long rx, tx;
            if (!colon)
                continue;
            *colon = 0;
            while (*name == ' ')
                name++;
            int k = !strcmp(name, NET_IF) ? 0 : !strcmp(name, TS_IF) ? 2 : -1;
            if (k < 0 || sscanf(colon + 1, "%llu %*u %*u %*u %*u %*u %*u %*u %llu", &rx, &tx) != 2)
                continue;
            if (have && dt > 0) {
                double r = rx >= last_net[k] ? (rx - last_net[k]) / dt : 0;
                double w = tx >= last_net[k + 1] ? (tx - last_net[k + 1]) / dt : 0;
                if (k == 0)
                    s->net_rx = r, s->net_tx = w;
                else
                    s->ts_rx = r, s->ts_tx = w;
            }
            last_net[k] = rx;
            last_net[k + 1] = tx;
        }
        fclose(f);
    }

    /* Major page faults */
    if ((f = fopen("/proc/vmstat", "r"))) {
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "pgmajfault ", 11))
                continue;
            unsigned long long m = strtoull(line + 11, NULL, 10);
            if (have && dt > 0 && m >= last_majflt)
                s->majflt = (m - last_majflt) / dt;
            last_majflt = m;
            break;
        }
        fclose(f);
    }

    /* VRAM (NVML) */
    for (int i = 0; nvml_ok && i < N_GPUS; i++) {
        nvmlMemory_t mem;
        if (nvml_dev[i] && nvmlDeviceGetMemoryInfo(nvml_dev[i], &mem) == NVML_SUCCESS && mem.total)
            s->vram[i] = (double)mem.used / mem.total;
    }
    last_t = t;
    have = 1;
}

/* ---- end system sensors ---- */

/* ---------------------------------------------------------------- simulated data */

static double frand(void) { return rand() / (double)RAND_MAX; }
static double ramp(double u, double a, double b) { return clamp01((u - a) / (b - a)); }

/* --demo: every value wanders through quiet, busy and flat out on its own rhythm */
static void demo_poll(stats *g, sys_stats *s, double t)
{
    for (int i = 0; i < N_GPUS; i++) {
        double a = clamp01(0.45 + 0.75 * sin(t * (0.11 + 0.04 * i) + i * 2.2));
        a = a < 0.12 ? 0 : a;
        g->load[i] = clamp01(a + (a > 0 ? (frand() - 0.5) * 0.06 : 0));
        g->power[i] = (i ? 32 : 24) + 520 * pow(a, 1.2);
        g->temp[i] = (int)(34 + 48 * a);
        g->tok_port[i] = a * (i ? 760 : 900);
        g->running_port[i] = a > 0 ? 1 + (int)(a * 5) : 0;
        s->vram[i] = a > 0 ? 0.35 + 0.55 * clamp01(0.5 + 0.5 * sin(t * 0.05 + i)) : 0.08 + 0.1 * i;
    }
    g->tok_s = g->tok_port[0] + g->tok_port[1];
    g->running = g->running_port[0] + g->running_port[1];

    double busy = clamp01(0.5 + 0.6 * sin(t * 0.17) + 0.2 * sin(t * 0.41));
    s->n_threads = 32;
    s->cpu_load = clamp01(0.03 + 0.95 * busy * busy + (frand() - 0.5) * 0.04);
    s->busy_threads = (int)(s->cpu_load * 32);
    for (int i = 0; i < 32; i++)
        s->thread_load[i] = clamp01(s->cpu_load * (0.6 + 0.8 * frand()));
    s->cpu_temp = 42 + 50 * pow(busy, 1.2);
    s->ram_total = 91.9;
    s->ram_used = 12 + 72 * clamp01(0.5 + 0.5 * sin(t * 0.07 - 1));
    double dk = 0.5 + 0.5 * sin(t * 0.23 + 1.3) * sin(t * 0.07);
    s->disk = dk > 0.45 ? pow(10, 5 + 4.4 * (dk - 0.45) / 0.55) : 2e4;
    double net = 0.5 + 0.5 * sin(t * 0.29 + 0.5);
    s->net_rx = pow(10, 2.5 + 5.5 * net);
    s->net_tx = pow(10, 2.5 + 3.5 * net);
    double tsn = 0.5 + 0.5 * sin(t * 0.19 + 2.5);
    s->ts_rx = tsn > 0.5 ? pow(10, 3 + 4.5 * (tsn - 0.5) * 2) : 300;
    s->ts_tx = s->ts_rx * 0.3;
    s->ctxt = pow(10, 4.2 + 2.0 * clamp01(0.5 + 0.5 * sin(t * 0.13 + 0.7)));
    s->majflt = fmod(t, 23) < 1.2 ? 9000 : 20;
    s->uptime = 11 * 86400 + 3600 * 5 + t;
}

/*
 * --showcase: a scripted 36 s arc for filming or GIFs. Night, all asleep; a download
 * wakes the radio and the disks wake the cat; GPU 0 starts, then GPU 1, the CPU
 * heats up and everything goes flat out (roaring fire, whistling teapot, frantic
 * cat, a full basket, VRAM filling, a burst of page faults drops a stitch); it winds
 * down and everyone goes back to sleep.
 */
#define SHOWCASE_LEN 36.0

static void showcase_poll(stats *g, sys_stats *s, double t)
{
    double u = fmod(t, SHOWCASE_LEN);
    double a0 = u < 7 ? 0 : u < 12 ? 0.55 * ramp(u, 7, 8.5) : u < 24 ? 0.55 + 0.4 * ramp(u, 12, 14)
              : u < 29 ? 0.95 - 0.6 * ramp(u, 24, 26) : 0.35 * (1 - ramp(u, 29, 30));
    double a1 = u < 11 ? 0 : u < 24 ? 0.9 * ramp(u, 11, 13) : 0.9 * (1 - ramp(u, 24, 25.5));
    double a[2] = { a0, a1 };
    for (int i = 0; i < N_GPUS; i++) {
        double j = a[i] > 0.01 ? 1 + (frand() - 0.5) * 0.06 : 0;
        g->tok_port[i] = a[i] * 900 * j;
        g->load[i] = clamp01(a[i] * 1.05);
        g->power[i] = 28 + 530 * a[i];
        g->temp[i] = (int)(36 + 46 * clamp01(a[i] * 1.05));
        g->running_port[i] = a[i] > 0.02 ? 1 + (int)(a[i] * 4) : 0;
        s->vram[i] = 0.1 + 0.8 * ramp(u, 7 + 4 * i, 20 + 2 * i) * (1 - ramp(u, 26, 32));
    }
    g->tok_s = g->tok_port[0] + g->tok_port[1];
    g->running = g->running_port[0] + g->running_port[1];

    double cpu = u < 5 ? 0.03 : u < 14 ? 0.03 + 0.4 * ramp(u, 5, 12) : u < 24 ? 0.45 + 0.52 * ramp(u, 14, 17)
               : 0.97 - 0.94 * ramp(u, 24, 30);
    s->n_threads = 32;
    s->cpu_load = clamp01(cpu + (frand() - 0.5) * 0.03);
    s->busy_threads = (int)(s->cpu_load * 32);
    for (int i = 0; i < 32; i++)
        s->thread_load[i] = s->cpu_load;
    s->cpu_temp = 41 + 52 * ramp(u, 9, 20) * (1 - ramp(u, 25, 33));
    s->ram_total = 91.9;
    s->ram_used = 10 + 76 * ramp(u, 8, 20) * (1 - 0.8 * ramp(u, 26, 31));
    double disk = u < 3.5 ? 0 : u < 7 ? 0.35 : u < 14 ? 0.2 : u < 23 ? 0.6 + 0.4 * ramp(u, 14, 17) : u < 26 ? 0.3 : 0;
    s->disk = disk > 0 ? pow(10, 5 + 4.4 * disk) : 1e4;
    double net = u < 2.5 ? 0 : u < 9 ? ramp(u, 2.5, 4) : u < 26 ? 0.6 : 0.6 * (1 - ramp(u, 26, 29));
    s->net_rx = pow(10, 2.5 + 5.5 * net);
    s->net_tx = pow(10, 2.5 + 3 * net);
    double tsn = u > 15 && u < 25 ? 0.8 : 0;
    s->ts_rx = pow(10, 2.5 + 5 * tsn);
    s->ts_tx = s->ts_rx * 0.4;
    s->ctxt = pow(10, 4.0 + 2.3 * (u < 13 ? 0.15 : u < 18 ? 0.2 + 0.8 * ramp(u, 13, 16) : u < 24 ? 1.0 : 0.15));
    s->majflt = (u > 19 && u < 19.8) ? 12000 : 15;
    s->uptime = 11 * 86400 + 7200 + t;
}

/* ---------------------------------------------------------------- scene */

#define FPS_BUSY        24
#define FPS_IDLE        15

#define SCROLL_MAX      80.0        /* px/s the scarf spills at, flat out */
#define SCROLL_KNEE     600.0       /* tok/s at which it's ~63% of that */
#define ROW_H           3.4         /* one knitted row, px along the scarf */
#define MAJF_PER_DROP   6000.0      /* major page faults per dropped stitch */
#define DROP_GAP        4.0         /* s, at least this long between dropped stitches */
#define DOZE_AFTER      3.0         /* s of idle before she nods off */

/* Grandma: sprite canvas top-left, the rockers' pivot, her hands (awake, dozing), her head */
#define GM_X            17.0
#define GM_Y            132.0
static const double PIVOT[2] = { 150, 392 };
static const double HAND_AWAKE[2] = { 187, 233 }, HAND_DOZE[2] = { 184, 262 };
static const double HEAD[2] = { 158, 168 };

static const double BALL_POS[N_GPUS][2] = { { 96, 406 }, { 150, 434 } };
#define BALL_RMAX       21.0
#define BALL_RMIN       9.0

/* Sampler cloth (inside the frame baked into room.png) */
#define SAMP_X0         126.0
#define SAMP_X1         351.0
#define SAMP_Y0         41.0
#define SAMP_Y1         148.0

#define CAL_X           273.0
#define CAL_Y           162.0
#define CAL_W           64.0
#define CAL_H           70.0

#define TEAPOT_X        333.0
#define TEAPOT_Y        256.0
#define RADIO_X         384.0
#define RADIO_Y         96.0
#define BASKET_X        367.0
#define BASKET_Y        318.0
static const double FIRE_BASE[2] = { 437, 268 };

/* Scarf centreline from her lap, over her knees, down to the rug, round a loop and away */
static const double SCARF_PTS[][2] = {
    { 199, 266 }, { 208, 292 }, { 217, 315 }, { 233, 335 }, { 249, 350 }, { 254, 372 }, { 257, 396 },
    { 270, 416 }, { 298, 430 }, { 332, 427 }, { 357, 407 }, { 361, 381 }, { 343, 364 }, { 318, 368 },
    { 304, 390 }, { 306, 420 }, { 316, 448 }, { 324, 476 }, { 330, 510 },
};
#define N_SCARF_PTS     (int)(sizeof(SCARF_PTS) / sizeof(SCARF_PTS[0]))
#define SCARF_ROCK_END  4           /* control points up to here rock with the chair */

typedef struct {
    double rate, rate_gpu[N_GPUS];          /* tok/s (token-equivalent in GPU mode) */
    double gtemp[N_GPUS], vram[N_GPUS];
    double cpu_load, cpu_temp, ram;
    double disk, lan, ts, ctxt;             /* 0..1 levels */
} view_t;

static cairo_surface_t *bg, *hud, *gm_img[2], *cat_img[3], *teapot_img, *radio_img, *basket_img, *yarn_img;
static cairo_surface_t *ball_tint[N_GPUS], *basket_ball[9], *puff_img, *note_img[2][2], *hand_mask, *flame_mask;
static char hud_key[160];
#define FLAME_MIN       3
#define N_FLAME_SPR     18
static cairo_surface_t *flame_spr[N_FLAME_SPR];     /* soft flame blobs, 3..20 px wide, twice as tall */

static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/knit/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/knit/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "knit: can't load %s (looked next to the binary and in ./assets/knit)\n", name);
        exit(1);
    }
    return s;
}

static int img_w(cairo_surface_t *s) { return cairo_image_surface_get_width(s); }
static int img_h(cairo_surface_t *s) { return cairo_image_surface_get_height(s); }

/* Yarn colours: blue for GPU 0, orange for GPU 1, warming towards plum / red with temperature */
static rgb yarn_color(int i, double temp)
{
    static const rgb cool[2] = { { 0.33, 0.56, 0.86 }, { 0.95, 0.58, 0.22 } };
    static const rgb warm[2] = { { 0.58, 0.40, 0.76 }, { 0.87, 0.30, 0.20 } };
    return lerp(cool[i], warm[i], clamp01((temp - 40) / 45));
}

/* Multiply the grey yarn ball by a colour, at size d */
static void tint_ball(cairo_surface_t *dst, rgb c)
{
    int d = img_w(dst);
    cairo_t *cr = cairo_create(dst);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_scale(cr, d / (double)img_w(yarn_img), d / (double)img_h(yarn_img));
    cairo_set_source_surface(cr, yarn_img, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_GOOD);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_MULTIPLY);
    cairo_set_source_rgb(cr, fmin(1, c.r * 1.25), fmin(1, c.g * 1.25), fmin(1, c.b * 1.25));
    cairo_mask_surface(cr, yarn_img, 0, 0);
    cairo_destroy(cr);
}

/* ---------------------------------------------------------------- cross-stitch font */

typedef struct { char c; unsigned char w, rows[7]; } glyph;
static const glyph FONT[] = {
    { '0', 5, { 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E } }, { '1', 5, { 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E } },
    { '2', 5, { 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F } }, { '3', 5, { 0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E } },
    { '4', 5, { 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 } }, { '5', 5, { 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E } },
    { '6', 5, { 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E } }, { '7', 5, { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 } },
    { '8', 5, { 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E } }, { '9', 5, { 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C } },
    { 'A', 5, { 0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 } }, { 'B', 5, { 0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E } },
    { 'C', 5, { 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E } }, { 'D', 5, { 0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C } },
    { 'E', 5, { 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F } }, { 'G', 5, { 0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F } },
    { 'H', 5, { 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 } }, { 'I', 3, { 0x07, 0x02, 0x02, 0x02, 0x02, 0x02, 0x07 } },
    { 'K', 5, { 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11 } }, { 'L', 5, { 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F } },
    { 'M', 5, { 0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11 } }, { 'N', 5, { 0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11 } },
    { 'O', 5, { 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E } }, { 'P', 5, { 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 } },
    { 'R', 5, { 0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11 } }, { 'S', 5, { 0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E } },
    { 'T', 5, { 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 } }, { 'U', 5, { 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E } },
    { 'Z', 5, { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F } }, { '%', 5, { 0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03 } },
    { '/', 5, { 0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10 } }, { ',', 2, { 0x00, 0x00, 0x00, 0x00, 0x03, 0x01, 0x02 } },
    { '.', 2, { 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x03 } }, { ' ', 2, { 0 } },
};

static const glyph *find_glyph(char c)
{
    for (unsigned i = 0; i < sizeof(FONT) / sizeof(FONT[0]); i++)
        if (FONT[i].c == c)
            return &FONT[i];
    return &FONT[sizeof(FONT) / sizeof(FONT[0]) - 1];
}

static double xs_width(const char *s, double cell)
{
    int cells = 0;
    for (const char *p = s; *p; p++)
        cells += find_glyph(*p)->w + 1;
    return (cells - 1) * cell;
}

/* One cross stitch: a shadowed X of thread, top leg lighter like real floss */
static void xs_cell(cairo_t *cr, double x, double y, double cell, rgb c)
{
    double m = cell * 0.14, lw = fmax(1.0, cell * 0.36);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(cr, lw + 0.9);
    cairo_set_source_rgba(cr, c.r * 0.35, c.g * 0.35, c.b * 0.35, 0.55);
    cairo_move_to(cr, x + m, y + m + 0.5); cairo_line_to(cr, x + cell - m, y + cell - m + 0.5);
    cairo_move_to(cr, x + cell - m, y + m + 0.5); cairo_line_to(cr, x + m, y + cell - m + 0.5);
    cairo_stroke(cr);
    cairo_set_line_width(cr, lw);
    set_rgb(cr, c);
    cairo_move_to(cr, x + m, y + m); cairo_line_to(cr, x + cell - m, y + cell - m);
    cairo_stroke(cr);
    cairo_set_source_rgb(cr, fmin(1, c.r * 1.25 + 0.06), fmin(1, c.g * 1.25 + 0.06), fmin(1, c.b * 1.25 + 0.06));
    cairo_move_to(cr, x + cell - m, y + m); cairo_line_to(cr, x + m, y + cell - m);
    cairo_stroke(cr);
}

/* Stitched glyphs are cached as small images, so a changing number costs a few blits */
typedef struct { char c; float cell; rgb col; cairo_surface_t *s; } glyph_img;
static glyph_img glyph_cache[64];
static int n_glyph_cache;

static cairo_surface_t *glyph_surface(char c, double cell, rgb col)
{
    for (int i = 0; i < n_glyph_cache; i++) {
        glyph_img *g = &glyph_cache[i];
        if (g->c == c && g->cell == (float)cell && g->col.r == col.r && g->col.g == col.g && g->col.b == col.b)
            return g->s;
    }
    if (n_glyph_cache == 64) {
        for (int i = 0; i < n_glyph_cache; i++)
            cairo_surface_destroy(glyph_cache[i].s);
        n_glyph_cache = 0;
    }
    const glyph *g = find_glyph(c);
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, (int)ceil(g->w * cell) + 4, (int)ceil(7 * cell) + 4);
    cairo_t *cr = cairo_create(s);
    for (int r = 0; r < 7; r++)
        for (int k = 0; k < g->w; k++)
            if (g->rows[r] >> (g->w - 1 - k) & 1)
                xs_cell(cr, 2 + k * cell, 2 + r * cell, cell, col);
    cairo_destroy(cr);
    glyph_cache[n_glyph_cache++] = (glyph_img){ c, (float)cell, col, s };
    return s;
}

static void xs_text(cairo_t *cr, const char *s, double cx, double top, double cell, rgb c)
{
    double x = cx - xs_width(s, cell) / 2;
    for (const char *p = s; *p; p++) {
        const glyph *g = find_glyph(*p);
        if (*p != ' ') {
            cairo_set_source_surface(cr, glyph_surface(*p, cell, c), round(x) - 2, round(top) - 2);
            cairo_paint(cr);
        }
        x += (g->w + 1) * cell;
    }
}

/* ---------------------------------------------------------------- static layers */

static void center_text(cairo_t *cr, double x, double y, double size, const char *s, const char *face)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, face, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    cairo_show_text(cr, s);
}

static void build_static(void)
{
    cairo_surface_t *room = load_asset("room.png");
    gm_img[0] = load_asset("grandma_knit.png");
    gm_img[1] = load_asset("grandma_doze.png");
    cat_img[0] = load_asset("cat_crouch.png");
    cat_img[1] = load_asset("cat_pounce.png");
    cat_img[2] = load_asset("cat_sleep.png");
    teapot_img = load_asset("teapot.png");
    radio_img = load_asset("radio.png");
    basket_img = load_asset("basket.png");
    yarn_img = load_asset("yarn.png");

    bg = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(bg);
    cairo_set_source_surface(cr, room, 0, 0);
    cairo_paint(cr);
    cairo_surface_destroy(room);

    /* sampler: soft inner shade and a stitched border */
    {
        cairo_pattern_t *p = cairo_pattern_create_radial(238, 95, 30, 238, 95, 150);
        cairo_pattern_add_color_stop_rgba(p, 0, 1, 0.97, 0.9, 0.10);
        cairo_pattern_add_color_stop_rgba(p, 1, 0.25, 0.12, 0.05, 0.22);
        cairo_rectangle(cr, SAMP_X0, SAMP_Y0, SAMP_X1 - SAMP_X0, SAMP_Y1 - SAMP_Y0);
        cairo_set_source(cr, p);
        cairo_fill(cr);
        cairo_pattern_destroy(p);
        const double cell = 3.0;
        rgb red = { 0.62, 0.12, 0.13 };
        int nx = (int)((SAMP_X1 - SAMP_X0 - 8) / cell), ny = (int)((SAMP_Y1 - SAMP_Y0 - 8) / cell);
        double x0 = (SAMP_X0 + SAMP_X1) / 2 - nx * cell / 2, y0 = (SAMP_Y0 + SAMP_Y1) / 2 - ny * cell / 2;
        for (int i = 0; i < nx; i++)
            if (i % 2 == 0) {
                xs_cell(cr, x0 + i * cell, y0, cell, red);
                xs_cell(cr, x0 + i * cell, y0 + (ny - 1) * cell, cell, red);
            }
        for (int j = 2; j < ny - 1; j++)
            if (j % 2 == 0) {
                xs_cell(cr, x0, y0 + j * cell, cell, red);
                xs_cell(cr, x0 + (nx - 1) * cell, y0 + j * cell, cell, red);
            }
    }

    /* wall calendar: shadow, two pages peeking below, the page, red header, rings */
    {
        cairo_set_source_rgba(cr, 0.1, 0.05, 0.02, 0.35);
        cairo_rectangle(cr, CAL_X + 3, CAL_Y + 4, CAL_W, CAL_H);
        cairo_fill(cr);
        for (int k = 2; k >= 1; k--) {
            cairo_set_source_rgb(cr, 0.86 - k * 0.05, 0.82 - k * 0.05, 0.72 - k * 0.05);
            cairo_rectangle(cr, CAL_X + k * 0.8, CAL_Y + k * 1.6, CAL_W, CAL_H);
            cairo_fill(cr);
        }
        cairo_pattern_t *p = cairo_pattern_create_linear(0, CAL_Y, 0, CAL_Y + CAL_H);
        cairo_pattern_add_color_stop_rgb(p, 0, 0.97, 0.93, 0.83);
        cairo_pattern_add_color_stop_rgb(p, 1, 0.88, 0.82, 0.70);
        cairo_rectangle(cr, CAL_X, CAL_Y, CAL_W, CAL_H);
        cairo_set_source(cr, p);
        cairo_fill(cr);
        cairo_pattern_destroy(p);
        cairo_rectangle(cr, CAL_X, CAL_Y, CAL_W, 25);
        cairo_set_source_rgb(cr, 0.66, 0.16, 0.14);
        cairo_fill(cr);
        for (int k = 0; k < 2; k++) {
            double rx = CAL_X + 16 + k * 32;
            cairo_arc(cr, rx, CAL_Y + 1, 3.2, M_PI, 2 * M_PI);
            cairo_set_line_width(cr, 1.6);
            cairo_set_source_rgb(cr, 0.25, 0.22, 0.2);
            cairo_stroke(cr);
        }
        cairo_arc(cr, CAL_X + CAL_W / 2, CAL_Y - 6, 1.8, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.3, 0.25, 0.2);
        cairo_fill(cr);
        cairo_move_to(cr, CAL_X + 16, CAL_Y);
        cairo_line_to(cr, CAL_X + CAL_W / 2, CAL_Y - 6);
        cairo_line_to(cr, CAL_X + 48, CAL_Y);
        cairo_set_line_width(cr, 0.8);
        cairo_stroke(cr);
        cairo_set_source_rgb(cr, 1, 0.95, 0.88);
        center_text(cr, CAL_X + CAL_W / 2, CAL_Y + 13.5, 22, "DAY", "DejaVu Serif");
    }

    /* props that don't move: the teapot on the hearth, the radio on the mantel */
    cairo_set_source_rgba(cr, 0.05, 0.02, 0.0, 0.35);
    cairo_save(cr);
    cairo_translate(cr, TEAPOT_X + 24, TEAPOT_Y + 33);
    cairo_scale(cr, 1, 0.25);
    cairo_arc(cr, 0, 0, 22, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_fill(cr);
    cairo_set_source_surface(cr, teapot_img, TEAPOT_X, TEAPOT_Y);
    cairo_paint(cr);
    cairo_set_source_surface(cr, radio_img, RADIO_X, RADIO_Y);
    cairo_paint(cr);
    cairo_destroy(cr);

    hud = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);

    /* yarn balls: tinted per GPU on the fly; basket balls in fixed colours */
    for (int i = 0; i < N_GPUS; i++)
        ball_tint[i] = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 48, 48);
    static const rgb stash[9] = {
        { 0.80, 0.30, 0.35 }, { 0.45, 0.62, 0.35 }, { 0.93, 0.78, 0.40 }, { 0.55, 0.40, 0.70 }, { 0.35, 0.60, 0.70 },
        { 0.92, 0.60, 0.50 }, { 0.70, 0.55, 0.35 }, { 0.85, 0.85, 0.80 }, { 0.40, 0.45, 0.75 },
    };
    for (int i = 0; i < 9; i++) {
        basket_ball[i] = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 24, 24);
        tint_ball(basket_ball[i], stash[i]);
    }

    /* a steam puff */
    puff_img = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 32, 32);
    cr = cairo_create(puff_img);
    cairo_pattern_t *p = cairo_pattern_create_radial(16, 16, 0, 16, 16, 16);
    cairo_pattern_add_color_stop_rgba(p, 0, 1, 0.98, 0.95, 0.9);
    cairo_pattern_add_color_stop_rgba(p, 0.5, 1, 0.97, 0.93, 0.45);
    cairo_pattern_add_color_stop_rgba(p, 1, 1, 0.97, 0.93, 0);
    cairo_set_source(cr, p);
    cairo_paint(cr);
    cairo_pattern_destroy(p);
    cairo_destroy(cr);

    /* music notes: [network][glyph], gold for the LAN, sky blue for tailscale */
    static const rgb note_col[2] = { { 1.0, 0.84, 0.42 }, { 0.55, 0.85, 1.0 } };
    static const char *glyphs[2] = { "♪", "♫" };
    for (int n = 0; n < 2; n++)
        for (int k = 0; k < 2; k++) {
            note_img[n][k] = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 32, 32);
            cr = cairo_create(note_img[n][k]);
            cairo_text_extents_t ext;
            cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
            cairo_set_font_size(cr, 22);
            cairo_text_extents(cr, glyphs[k], &ext);
            cairo_move_to(cr, 16 - ext.width / 2 - ext.x_bearing, 16 - ext.height / 2 - ext.y_bearing);
            cairo_text_path(cr, glyphs[k]);
            cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
            cairo_set_line_width(cr, 3);
            cairo_set_source_rgba(cr, 0.15, 0.07, 0.02, 0.75);
            cairo_stroke_preserve(cr);
            set_rgb(cr, note_col[n]);
            cairo_fill(cr);
            cairo_destroy(cr);
        }

    /* a soft blob of flame */
    flame_mask = cairo_image_surface_create(CAIRO_FORMAT_A8, 32, 32);
    cr = cairo_create(flame_mask);
    p = cairo_pattern_create_radial(16, 16, 0, 16, 16, 16);
    cairo_pattern_add_color_stop_rgba(p, 0, 0, 0, 0, 1);
    cairo_pattern_add_color_stop_rgba(p, 0.4, 0, 0, 0, 0.7);
    cairo_pattern_add_color_stop_rgba(p, 1, 0, 0, 0, 0);
    cairo_set_source(cr, p);
    cairo_paint(cr);
    cairo_pattern_destroy(p);
    cairo_destroy(cr);

    for (int i = 0; i < N_FLAME_SPR; i++) {
        int w = FLAME_MIN + i, h = 2 * w;
        flame_spr[i] = cairo_image_surface_create(CAIRO_FORMAT_A8, w, h);
        cr = cairo_create(flame_spr[i]);
        cairo_scale(cr, w / 32.0, h / 32.0);
        cairo_set_source_surface(cr, flame_mask, 0, 0);
        cairo_paint(cr);
        cairo_destroy(cr);
    }

    /* feathered disc around her hands, for the needle click */
    hand_mask = cairo_image_surface_create(CAIRO_FORMAT_A8, 80, 80);
    cr = cairo_create(hand_mask);
    p = cairo_pattern_create_radial(40, 40, 18, 40, 40, 36);
    cairo_pattern_add_color_stop_rgba(p, 0, 0, 0, 0, 1);
    cairo_pattern_add_color_stop_rgba(p, 1, 0, 0, 0, 0);
    cairo_set_source(cr, p);
    cairo_paint(cr);
    cairo_pattern_destroy(p);
    cairo_destroy(cr);
}

/* ---------------------------------------------------------------- scarf */

#define PATH_CAP    1400

static double path_x[PATH_CAP], path_y[PATH_CAP];     /* centreline every 1 px */
static int    path_n, path_rock_n;                     /* samples; samples that rock with the chair */

static void build_path(void)
{
    /* Catmull-Rom through the control points, then resampled at 1 px arc length */
    static double dx[PATH_CAP * 8], dy[PATH_CAP * 8];
    int nd = 0, rock_nd = 0;
    for (int i = 0; i < N_SCARF_PTS - 1; i++) {
        const double *p0 = SCARF_PTS[i > 0 ? i - 1 : 0], *p1 = SCARF_PTS[i], *p2 = SCARF_PTS[i + 1];
        const double *p3 = SCARF_PTS[i + 2 < N_SCARF_PTS ? i + 2 : N_SCARF_PTS - 1];
        for (int k = 0; k < 40 && nd < PATH_CAP * 8; k++) {
            double t = k / 40.0, t2 = t * t, t3 = t2 * t;
            for (int c = 0; c < 2; c++) {
                double v = 0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2 +
                                  (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3);
                (c ? dy : dx)[nd] = v;
            }
            nd++;
        }
        if (i + 1 == SCARF_ROCK_END)
            rock_nd = nd;
    }
    dx[nd] = SCARF_PTS[N_SCARF_PTS - 1][0];
    dy[nd] = SCARF_PTS[N_SCARF_PTS - 1][1];
    nd++;
    double acc = 0, next = 0;
    path_n = 0;
    for (int i = 1; i < nd && path_n < PATH_CAP; i++) {
        double seg = hypot(dx[i] - dx[i - 1], dy[i] - dy[i - 1]);
        while (next <= acc + seg && path_n < PATH_CAP) {
            double u = seg > 0 ? (next - acc) / seg : 0;
            path_x[path_n] = dx[i - 1] + (dx[i] - dx[i - 1]) * u;
            path_y[path_n] = dy[i - 1] + (dy[i] - dy[i - 1]) * u;
            path_n++;
            next += 1;
        }
        acc += seg;
        if (i == rock_nd)
            path_rock_n = path_n;
    }
}

typedef struct { float r, g, b; signed char ladder; } row_t;

#define MAX_ROWS    420
static row_t  rows[MAX_ROWS];                  /* rows[0] is the newest, at her lap */
static long   rows_total;                      /* rows knitted so far (the scarf's length) */
static double scroll_phase;                    /* 0..ROW_H: how far the newest row has emerged */
static double stitches;                        /* tokens so far */
static int    stripe_yarn, stripe_left, stripe_shade;
static double stripe_err[N_GPUS];
static int    ladder_left, ladder_col;

static rgb stripe_col;

static void push_row(const view_t *v)
{
    if (stripe_left <= 0) {
        /* New stripe: the yarn whose GPU is owed the most rows (error diffusion by share) */
        double tot = v->rate_gpu[0] + v->rate_gpu[1];
        int prev = stripe_yarn;
        for (int i = 0; i < N_GPUS; i++)
            stripe_err[i] += tot > 0 ? v->rate_gpu[i] / tot : 0.5;
        stripe_yarn = stripe_err[1] > stripe_err[0] ? 1 : 0;
        stripe_err[stripe_yarn] -= 1;
        /* the same yarn twice running: a darker dye lot, so the stripe still shows */
        stripe_shade = stripe_yarn == prev ? !stripe_shade : 0;
        rgb c = yarn_color(stripe_yarn, v->gtemp[stripe_yarn]);
        double k = stripe_shade ? 0.74 : 1.0;
        stripe_col = (rgb){ c.r * k, c.g * k, c.b * k };
        /* busier scheduler (context switches) = she changes yarn more often */
        stripe_left = (int)lround(14 - 11 * v->ctxt);
    }
    stripe_left--;
    memmove(rows + 1, rows, sizeof(rows[0]) * (MAX_ROWS - 1));
    rows[0] = (row_t){ (float)stripe_col.r, (float)stripe_col.g, (float)stripe_col.b, -1 };
    if (ladder_left > 0) {
        rows[0].ladder = (signed char)ladder_col;
        ladder_left--;
    }
    rows_total++;
}

/* Rocking: the part of the scarf on her lap moves with the chair, fading out towards the rug */
static void rock_point(double x, double y, double ang, double *ox, double *oy)
{
    double c = cos(ang), s = sin(ang), dx = x - PIVOT[0], dy = y - PIVOT[1];
    *ox = PIVOT[0] + dx * c - dy * s;
    *oy = PIVOT[1] + dx * s + dy * c;
}

static double scarf_w(double s) { return 29 - 3 * clamp01((s - 80) / 60); }

/* Point and unit normal on the (unrocked) centreline at arc length s */
static void path_at(double s, double *x, double *y, double *nx, double *ny)
{
    if (s < 0)
        s = 0;
    if (s > path_n - 2)
        s = path_n - 2;
    int i = (int)s;
    double f = s - i;
    *x = path_x[i] + (path_x[i + 1] - path_x[i]) * f;
    *y = path_y[i] + (path_y[i + 1] - path_y[i]) * f;
    int a = i > 2 ? i - 2 : 0, b = i + 3 < path_n ? i + 3 : path_n - 1;
    double tx = path_x[b] - path_x[a], ty = path_y[b] - path_y[a], l = hypot(tx, ty);
    if (l < 1e-6)
        l = 1;
    *nx = -ty / l;
    *ny = tx / l;
}

/*
 * The scarf is drawn in two layers: the part on her lap (rocks with the chair) and
 * the part hanging to the rug and lying on it (still). Each frame only the flat yarn
 * colours are filled, row by row in runs of one colour; the knitted texture (stitches,
 * sheen, edges) is pre-rendered for SCARF_PHASES sub-row positions and laid over the
 * colours with ATOP, so the stitches scroll with the yarn at almost no cost.
 */
#define SCARF_K         5           /* stitch columns across */
#define STITCH_TW       24.0        /* the stitch tile: one stitch, drawn big */
#define STITCH_TH       14.0
#define SCARF_PHASES    8

typedef struct {
    double s_from, s_to;                        /* arc length this layer covers */
    int    x, y, w, h;                          /* where it sits on screen */
    cairo_surface_t *col, *tex[SCARF_PHASES];
} scarf_layer;

static int scarf_shadow_baked;
static scarf_layer layers[2];                   /* 0: lap, 1: floor */
static cairo_pattern_t *stitch_pat;

typedef struct { double x0, y0, nx0, ny0, x1, y1, nx1, ny1, w0, w1; } row_geom;

static int row_geometry(double s0, double s1, double from, double to, row_geom *q)
{
    s0 = fmax(s0, from);
    s1 = fmin(s1, to);
    if (s1 <= s0)
        return 0;
    path_at(s0, &q->x0, &q->y0, &q->nx0, &q->ny0);
    path_at(s1 + 0.35, &q->x1, &q->y1, &q->nx1, &q->ny1);   /* overlap a hair: no seams */
    q->w0 = scarf_w(s0) / 2;
    q->w1 = scarf_w(s1) / 2;
    return 1;
}

static void quad_path(cairo_t *cr, const row_geom *q)
{
    cairo_move_to(cr, q->x0 + q->nx0 * q->w0, q->y0 + q->ny0 * q->w0);
    cairo_line_to(cr, q->x1 + q->nx1 * q->w1, q->y1 + q->ny1 * q->w1);
    cairo_line_to(cr, q->x1 - q->nx1 * q->w1, q->y1 - q->ny1 * q->w1);
    cairo_line_to(cr, q->x0 - q->nx0 * q->w0, q->y0 - q->ny0 * q->w0);
    cairo_close_path(cr);
}

/* One knit stitch (a V) as shading over the yarn colour */
static void build_stitch(void)
{
    cairo_surface_t *tile = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, (int)STITCH_TW, (int)STITCH_TH);
    cairo_t *cr = cairo_create(tile);
    cairo_set_source_rgba(cr, 0.10, 0.04, 0.02, 0.45);
    cairo_paint(cr);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int pass = 0; pass < 2; pass++) {
        cairo_set_operator(cr, pass ? CAIRO_OPERATOR_OVER : CAIRO_OPERATOR_SOURCE);
        for (int dy = -1; dy <= 1; dy++) {            /* legs continue into the rows above and below */
            double oy = dy * STITCH_TH;
            cairo_move_to(cr, 1.5, oy - 1);
            cairo_line_to(cr, STITCH_TW / 2 - 0.5, oy + STITCH_TH - 1);
            cairo_move_to(cr, STITCH_TW - 1.5, oy - 1);
            cairo_line_to(cr, STITCH_TW / 2 + 0.5, oy + STITCH_TH - 1);
        }
        cairo_set_line_width(cr, pass ? 3.0 : 8.0);
        if (pass)
            cairo_set_source_rgba(cr, 1, 0.96, 0.88, 0.18);
        else
            cairo_set_source_rgba(cr, 0, 0, 0, 0);
        cairo_stroke(cr);
    }
    cairo_destroy(cr);
    stitch_pat = cairo_pattern_create_for_surface(tile);
    cairo_pattern_set_extend(stitch_pat, CAIRO_EXTEND_REPEAT);
    cairo_pattern_set_filter(stitch_pat, CAIRO_FILTER_GOOD);
    cairo_surface_destroy(tile);
}

/* Texture for one layer at one sub-row phase: rows oldest first, each erasing what it covers */
static void build_texture(scarf_layer *L, int ph)
{
    double phase = ph * ROW_H / SCARF_PHASES, len = path_n - 2;
    cairo_t *cr = cairo_create(L->tex[ph]);
    cairo_translate(cr, -L->x, -L->y);
    int nrows = (int)(len / ROW_H) + 3;
    for (int k = nrows; k >= 0; k--) {
        row_geom q;
        double s0 = phase + (k - 1) * ROW_H;
        if (!row_geometry(s0, s0 + ROW_H, fmax(0, L->s_from), fmin(len, L->s_to), &q))
            continue;
        quad_path(cr, &q);
        cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
        cairo_fill_preserve(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
        /* the tile, oriented along this row */
        double tx = q.ny0, ty = -q.nx0;
        double su = STITCH_TW / (2 * q.w0 / SCARF_K), sv = STITCH_TH / ROW_H;
        cairo_matrix_t m;
        cairo_matrix_init(&m, q.nx0 * su, tx * sv, q.ny0 * su, ty * sv,
                          (-q.x0 * q.nx0 - q.y0 * q.ny0 + q.w0) * su, (-q.x0 * tx - q.y0 * ty) * sv);
        cairo_matrix_translate(&m, L->x, L->y);
        cairo_pattern_set_matrix(stitch_pat, &m);
        cairo_set_source(cr, stitch_pat);
        cairo_fill(cr);
        /* sheen down the middle, darker edges */
        double mx0 = q.x0, my0 = q.y0, mx1 = q.x1, my1 = q.y1;
        cairo_move_to(cr, mx0, my0);
        cairo_line_to(cr, mx1, my1);
        cairo_set_line_width(cr, q.w0 * 0.9);
        cairo_set_source_rgba(cr, 1, 0.95, 0.85, 0.08);
        cairo_stroke(cr);
        for (int side = -1; side <= 1; side += 2) {
            cairo_move_to(cr, q.x0 + side * q.nx0 * (q.w0 - 1), q.y0 + side * q.ny0 * (q.w0 - 1));
            cairo_line_to(cr, q.x1 + side * q.nx1 * (q.w1 - 1), q.y1 + side * q.ny1 * (q.w1 - 1));
        }
        cairo_set_line_width(cr, 2.2);
        cairo_set_source_rgba(cr, 0.10, 0.04, 0.02, 0.45);
        cairo_stroke(cr);
    }
    cairo_destroy(cr);
}

static void build_scarf(void)
{
    double len = path_n - 2;
    layers[0].s_from = -1;
    layers[0].s_to = path_rock_n + 8;           /* the lap layer overlaps the knee a little */
    layers[1].s_from = path_rock_n - 2;
    layers[1].s_to = len + 1;
    for (int l = 0; l < 2; l++) {
        scarf_layer *L = &layers[l];
        double x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
        for (int i = (int)fmax(0, L->s_from); i <= (int)fmin(len, L->s_to); i++) {
            x0 = fmin(x0, path_x[i]); x1 = fmax(x1, path_x[i]);
            y0 = fmin(y0, path_y[i]); y1 = fmax(y1, path_y[i]);
        }
        L->x = (int)floor(x0) - 20;
        L->y = (int)floor(y0) - 20;
        L->w = (int)ceil(x1) + 20 - L->x;
        L->h = (int)ceil(y1) + 20 - L->y;
        L->col = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, L->w, L->h);
        for (int p = 0; p < SCARF_PHASES; p++) {
            L->tex[p] = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, L->w, L->h);
            build_texture(L, p);
        }
    }
}

/* Colours (and dropped-stitch ladders) of this frame's rows into one layer, textured */
static void paint_layer(scarf_layer *L, long nvis, double phase)
{
    double len = path_n - 2;
    cairo_t *cr = cairo_create(L->col);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_translate(cr, -L->x, -L->y);

    /* flat colours: oldest first, consecutive rows of one colour filled together */
    const row_t *run = NULL;
    int nrun = 0;
    for (long k = nvis - 1; k >= -1; k--) {
        row_geom q;
        const row_t *r = k >= 0 ? &rows[k] : NULL;
        double s0 = phase + (k - 1) * ROW_H;
        int have = r && row_geometry(s0, s0 + ROW_H, fmax(0, L->s_from), fmin(len, L->s_to), &q);
        double a = clamp01((fmax(0, s0) + 4) / 16);          /* fades in under her knitting */
        int single = have && a < 1;
        if (nrun && (!r || single || (have && (r->r != run->r || r->g != run->g || r->b != run->b)))) {
            cairo_set_source_rgb(cr, run->r, run->g, run->b);
            cairo_fill(cr);
            nrun = 0;
        }
        if (!have)
            continue;
        quad_path(cr, &q);
        if (single) {
            cairo_set_source_rgba(cr, r->r, r->g, r->b, a);
            cairo_fill(cr);
            continue;
        }
        run = r;
        nrun++;
    }

    /* the knitting itself */
    int p = (int)(phase / ROW_H * SCARF_PHASES) % SCARF_PHASES;
    cairo_identity_matrix(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_ATOP);
    cairo_set_source_surface(cr, L->tex[p], 0, 0);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_translate(cr, -L->x, -L->y);

    /* dropped stitches: a gap one column wide with the rung of yarn across it */
    for (long k = nvis - 1; k >= 0; k--) {
        const row_t *r = &rows[k];
        row_geom q;
        double s0 = phase + (k - 1) * ROW_H;
        if (r->ladder < 0 || !row_geometry(s0, s0 + ROW_H, fmax(0, L->s_from), fmin(len, L->s_to), &q))
            continue;
        double b0 = -q.w0 + (r->ladder + 0.1) * 2 * q.w0 / SCARF_K, b1 = -q.w0 + (r->ladder + 0.9) * 2 * q.w0 / SCARF_K;
        cairo_move_to(cr, q.x0 + q.nx0 * b0, q.y0 + q.ny0 * b0);
        cairo_line_to(cr, q.x0 + q.nx0 * b1, q.y0 + q.ny0 * b1);
        cairo_line_to(cr, q.x1 + q.nx1 * b1, q.y1 + q.ny1 * b1);
        cairo_line_to(cr, q.x1 + q.nx1 * b0, q.y1 + q.ny1 * b0);
        cairo_close_path(cr);
        cairo_set_source_rgba(cr, 0.10, 0.04, 0.03, 0.85);
        cairo_fill(cr);
        double mx = (q.x0 + q.x1) / 2, my = (q.y0 + q.y1) / 2;
        cairo_move_to(cr, mx + q.nx0 * b0, my + q.ny0 * b0);
        cairo_line_to(cr, mx + q.nx0 * b1, my + q.ny0 * b1);
        cairo_set_line_width(cr, 1.0);
        cairo_set_source_rgb(cr, fmin(1, r->r * 1.2), fmin(1, r->g * 1.2), fmin(1, r->b * 1.2));
        cairo_stroke(cr);
    }
    cairo_destroy(cr);
}

static void draw_scarf(cairo_t *cr, double rock)
{
    double len = path_n - 2;
    /* rows move in steps of 1/SCARF_PHASES of a row, in lockstep with the texture */
    double phase = floor(scroll_phase / ROW_H * SCARF_PHASES) * ROW_H / SCARF_PHASES;
    long nvis = (long)((len - phase) / ROW_H) + 2;
    if (nvis > rows_total)
        nvis = rows_total;
    if (nvis > MAX_ROWS)
        nvis = MAX_ROWS;
    double s_tail = fmin(len, phase + (nvis - 1) * ROW_H);

    /* shadow on the rug (baked into the lit room once the cast-on end has gone) */
    if (!scarf_shadow_baked) {
    cairo_save(cr);
    cairo_translate(cr, 2, 3.5);
    int i0 = (int)layers[1].s_from + 4;
    for (int i = i0; i <= (int)s_tail; i += 3)
        (i == i0 ? cairo_move_to : cairo_line_to)(cr, path_x[i], path_y[i]);
    cairo_set_line_width(cr, 28);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_source_rgba(cr, 0.08, 0.02, 0.0, 0.30);
    cairo_stroke(cr);
    cairo_restore(cr);
    }

    /* tassels on the cast-on end, if it is still in view */
    if (nvis == rows_total && s_tail < len - 2) {
        const row_t *r = &rows[nvis - 1];
        double x, y, nx, ny;
        path_at(s_tail + ROW_H, &x, &y, &nx, &ny);
        double w = scarf_w(s_tail);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        for (int k = 0; k < 7; k++) {
            double b = (k / 6.0 - 0.5) * (w - 5);
            double bx = x + nx * b, by = y + ny * b;
            double ex = bx + ny * 11 + nx * sin(k * 2.1) * 2, ey = by - nx * 11 + ny * sin(k * 2.1) * 2;
            cairo_move_to(cr, bx, by);
            cairo_line_to(cr, ex, ey);
            cairo_set_line_width(cr, 3.4);
            cairo_set_source_rgba(cr, r->r * 0.4, r->g * 0.4, r->b * 0.4, 0.8);
            cairo_stroke(cr);
            cairo_move_to(cr, bx, by);
            cairo_line_to(cr, ex, ey);
            cairo_set_line_width(cr, 2.2);
            cairo_set_source_rgb(cr, r->r, r->g, r->b);
            cairo_stroke(cr);
        }
    }

    paint_layer(&layers[1], nvis, phase);
    paint_layer(&layers[0], nvis, phase);
    cairo_set_source_surface(cr, layers[1].col, layers[1].x, layers[1].y);
    cairo_paint(cr);
    cairo_save(cr);
    cairo_translate(cr, PIVOT[0], PIVOT[1]);
    cairo_rotate(cr, rock);
    cairo_translate(cr, -PIVOT[0], -PIVOT[1]);
    cairo_set_source_surface(cr, layers[0].col, layers[0].x, layers[0].y);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);
    cairo_restore(cr);
}

/* ---------------------------------------------------------------- characters and props */

enum { CAT_SLEEP, CAT_CROUCH, CAT_POUNCE };

typedef struct {
    double x, y, fx, fy, tx, ty;        /* feet; pounce from / to */
    double t, dur, wait, height;
    double sleep;                       /* 0 awake .. 1 curled up */
    int    state, face;
} cat_t;

typedef struct { double x, y, vx, vy, age, life, size, spin; int kind, sub; } particle;

#define MAX_PARTS 160
static particle parts[MAX_PARTS];
enum { P_STEAM, P_NOTE, P_ZZZ, P_GLINT, P_LOOP };

#define MAX_FLAMES 110
static particle flames[MAX_FLAMES];
static double flame_acc;

static cat_t  cat;
static double ball_roll[N_GPUS], ball_r[N_GPUS], ball_kick[N_GPUS], ball_kick_v[N_GPUS];
static rgb    ball_col[N_GPUS];
static double doze, idle_for, busy_for, rock_t, rock_amp, needle_phase, click_flash;
static double steam_acc, note_acc[2], zzz_acc, drop_acc, since_drop = 99, fire_flick;
static int    dozing, basket_n, basket_target;
static double basket_scale[9];

static void spawn(int kind, int sub, double x, double y, double vx, double vy, double life, double size)
{
    for (int i = 0; i < MAX_PARTS; i++)
        if (parts[i].life <= 0) {
            parts[i] = (particle){ x, y, vx, vy, 0, life, size, frand() * 2 - 1, kind, sub };
            return;
        }
}

static void cat_init(void)
{
    cat = (cat_t){ .x = 300, .y = 440, .sleep = 1, .state = CAT_SLEEP, .face = 1 };
}

/* Somewhere to pounce: the yarn balls, the scarf, or a patch of rug, all inside the circle */
static void cat_pick_target(double chaos)
{
    static const double spots[][2] = {
        { 178, 440 }, { 205, 420 }, { 240, 446 }, { 272, 432 }, { 300, 442 }, { 326, 452 }, { 280, 410 },
    };
    int n = sizeof(spots) / sizeof(spots[0]);
    double best = -1, bx = cat.x, by = cat.y;
    for (int tries = 0; tries < 6; tries++) {
        int k = rand() % n;
        double x = spots[k][0] + (frand() - 0.5) * 16, y = spots[k][1] + (frand() - 0.5) * 8;
        double d = hypot(x - cat.x, y - cat.y);
        double want = 40 + 70 * chaos;                  /* calm hops are short */
        double score = -fabs(d - want) + ((x > cat.x) == (cat.face > 0) ? 25 : 0) + frand() * 10;
        if (d > 18 && score > best)
            best = score, bx = x, by = y;
    }
    cat.fx = cat.x, cat.fy = cat.y, cat.tx = bx, cat.ty = by;
    if (fabs(bx - cat.x) > 12)
        cat.face = bx > cat.x ? 1 : -1;
    cat.t = 0;
    cat.dur = 0.42 + hypot(bx - cat.x, by - cat.y) / 420;
    cat.height = 14 + 42 * chaos;
    cat.state = CAT_POUNCE;
}

static void cat_update(double chaos, int want_sleep, double dt)
{
    cat.t += dt;
    if (cat.state == CAT_SLEEP) {
        cat.sleep = fmin(1, cat.sleep + dt / 1.0);
        if (!want_sleep) {
            cat.state = CAT_CROUCH;
            cat.wait = 1.0;
            cat.t = 0;
        }
        return;
    }
    cat.sleep = fmax(0, cat.sleep - dt / 0.6);
    if (cat.state == CAT_CROUCH) {
        if (cat.t < (want_sleep ? fmin(cat.wait, 0.8) : cat.wait))
            return;
        if (want_sleep) {
            /* hop back onto the scarf to sleep */
            if (hypot(cat.x - 300, cat.y - 440) < 4) {
                cat.state = CAT_SLEEP;
                cat.t = 0;
                return;
            }
            cat.fx = cat.x, cat.fy = cat.y, cat.tx = 300, cat.ty = 440;
            if (fabs(300 - cat.x) > 12)
                cat.face = 300 > cat.x ? 1 : -1;
            cat.t = 0, cat.dur = 0.6, cat.height = 16, cat.state = CAT_POUNCE;
            return;
        }
        cat_pick_target(chaos);
        return;
    }
    /* in the air */
    double u = clamp01(cat.t / cat.dur);
    cat.x = cat.fx + (cat.tx - cat.fx) * u;
    cat.y = cat.fy + (cat.ty - cat.fy) * u;
    if (u >= 1 && want_sleep && hypot(cat.x - 300, cat.y - 440) < 4) {
        cat.state = CAT_SLEEP;                      /* curled up on the scarf */
        cat.t = 0;
    } else if (u >= 1) {
        cat.state = CAT_CROUCH;
        cat.t = 0;
        cat.wait = (3.2 - 2.8 * chaos) * (0.7 + 0.6 * frand());
        for (int i = 0; i < N_GPUS; i++)             /* landed by a yarn ball: bat it */
            if (hypot(cat.x - BALL_POS[i][0], cat.y - BALL_POS[i][1]) < 55)
                ball_kick_v[i] += (cat.face > 0 ? -1 : 1) * (40 + 60 * chaos) * (0.5 + frand());
    }
}

static void draw_sprite(cairo_t *cr, cairo_surface_t *img, double x, double y, double ax, double ay, int flip, double alpha)
{
    int w = img_w(img), h = img_h(img);
    cairo_save(cr);
    cairo_translate(cr, round(x), round(y));
    if (flip)
        cairo_scale(cr, -1, 1);
    cairo_translate(cr, -ax * w, -ay * h);
    cairo_set_source_surface(cr, img, 0, 0);
    if (alpha >= 1)
        cairo_paint(cr);
    else
        cairo_paint_with_alpha(cr, alpha);
    cairo_restore(cr);
}

static void draw_shadow(cairo_t *cr, double x, double y, double rx, double ry, double a)
{
    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, rx, ry);
    cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgba(cr, 0.06, 0.02, 0.0, a);
    cairo_fill(cr);
}

static void draw_cat(cairo_t *cr, double t)
{
    int flip = cat.face < 0;
    if (cat.state == CAT_POUNCE) {
        double u = clamp01(cat.t / cat.dur), lift = cat.height * sin(M_PI * u);
        draw_shadow(cr, cat.x, cat.y - 2, 24 - lift * 0.15, 5, 0.35 - lift * 0.004);
        if (u < 0.12 || u > 0.9)
            draw_sprite(cr, cat_img[0], cat.x, cat.y, 0.5, 0.97, flip, 1);
        else
            draw_sprite(cr, cat_img[1], cat.x, cat.y - lift + 6, 0.5, 0.97, flip, 1);
        return;
    }
    draw_shadow(cr, cat.x, cat.y - 2, 26, 5, 0.35);
    if (cat.sleep < 1) {
        /* wiggle before a pounce */
        double wig = cat.state == CAT_CROUCH && cat.wait - cat.t < 0.35 ? sin(t * 60) * 1.2 : 0;
        draw_sprite(cr, cat_img[0], cat.x + wig, cat.y, 0.5, 0.97, flip, 1 - cat.sleep);
    }
    if (cat.sleep > 0) {
        double breathe = 1 + 0.025 * sin(t * 2.2);
        cairo_save(cr);
        cairo_translate(cr, cat.x, cat.y);
        cairo_scale(cr, 1, breathe);
        draw_sprite(cr, cat_img[2], 0, 0, 0.5, 0.95, flip, cat.sleep);
        cairo_restore(cr);
    }
}

static void draw_fire(cairo_t *cr, double heat, double t)
{
    const double bx = FIRE_BASE[0], by = FIRE_BASE[1];

    /* a flickering glow at the mouth of the fireplace (the steady glow on the room is in lit) */
    double gr = 55 + 35 * heat;
    cairo_pattern_t *p = cairo_pattern_create_radial(bx, by - 18, 0, bx, by - 18, gr);
    cairo_pattern_add_color_stop_rgba(p, 0, 1.0, 0.55, 0.18, (0.10 + 0.16 * heat) * (0.5 + 0.5 * fire_flick));
    cairo_pattern_add_color_stop_rgba(p, 1, 1.0, 0.4, 0.1, 0);
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_arc(cr, bx, by - 18, gr, 0, 2 * M_PI);
    cairo_set_source(cr, p);
    cairo_fill(cr);
    cairo_pattern_destroy(p);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* flames: soft blobs rising off the logs, added up so they glow, clipped to the firebox */
    cairo_save(cr);
    cairo_move_to(cr, 396, 282);
    cairo_line_to(cr, 396, 214);
    cairo_curve_to(cr, 405, 190, 470, 184, 482, 196);
    cairo_line_to(cr, 482, 282);
    cairo_close_path(cr);
    cairo_clip(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = 0; i < MAX_FLAMES; i++) {
        const particle *f = &flames[i];
        if (f->life <= 0)
            continue;
        double u = f->age / f->life, d = f->size * (1 - 0.6 * u);
        rgb c = u < 0.3 ? lerp((rgb){ 1.0, 0.78, 0.35 }, (rgb){ 1.0, 0.48, 0.10 }, u / 0.3)
                        : lerp((rgb){ 1.0, 0.48, 0.10 }, (rgb){ 0.70, 0.10, 0.03 }, (u - 0.3) / 0.7);
        cairo_set_source_rgba(cr, c.r, c.g, c.b, (u < 0.15 ? u / 0.15 : 1 - (u - 0.15) / 0.85) * 0.42);
        int k = (int)lround(d) - FLAME_MIN;
        k = k < 0 ? 0 : k >= N_FLAME_SPR ? N_FLAME_SPR - 1 : k;
        cairo_mask_surface(cr, flame_spr[k], round(f->x - img_w(flame_spr[k]) / 2.0),
                           round(f->y - img_h(flame_spr[k]) / 2.0));
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_restore(cr);
    (void)t;
}

static void draw_basket(cairo_t *cr, double t)
{
    static const double slot[9][2] = {
        { 16, 11 }, { 30, 9 }, { 44, 10 }, { 58, 11 }, { 23, 1 }, { 37, 0 }, { 51, 2 }, { 30, -8 }, { 44, -8 },
    };
    for (int i = 0; i < 9; i++) {
        double k = basket_scale[i];
        if (k <= 0.01)
            continue;
        double d = 20 * k;
        cairo_save(cr);
        cairo_translate(cr, BASKET_X + slot[i][0] + 1, BASKET_Y + slot[i][1] + 2 + (1 - k) * 8);
        cairo_rotate(cr, i * 1.3 + 0.05 * sin(t + i));
        cairo_scale(cr, d / 24, d / 24);
        cairo_set_source_surface(cr, basket_ball[i], -12, -12);
        cairo_paint(cr);
        cairo_restore(cr);
    }
    draw_shadow(cr, BASKET_X + 38, BASKET_Y + 46, 36, 5, 0.3);
    cairo_set_source_surface(cr, basket_img, BASKET_X, BASKET_Y);
    cairo_paint(cr);
}

static void draw_ball(cairo_t *cr, int i, double temp)
{
    double x = BALL_POS[i][0] + ball_kick[i], y = BALL_POS[i][1], r = ball_r[i];
    double y0 = y + BALL_RMAX - r;                  /* sits on the rug */
    double warm = clamp01((temp - 40) / 45);
    /* warm glow behind it */
    double gr = r * 2.3;
    cairo_pattern_t *p = cairo_pattern_create_radial(x, y0, r * 0.6, x, y0, gr);
    cairo_pattern_add_color_stop_rgba(p, 0, 1.0, 0.55 - 0.25 * warm, 0.2, 0.15 + 0.45 * warm);
    cairo_pattern_add_color_stop_rgba(p, 1, 1.0, 0.4, 0.1, 0);
    cairo_arc(cr, x, y0, gr, 0, 2 * M_PI);
    cairo_set_source(cr, p);
    cairo_fill(cr);
    cairo_pattern_destroy(p);
    draw_shadow(cr, x + 2, y0 + r - 1, r * 1.05, r * 0.28, 0.45);
    cairo_save(cr);
    cairo_translate(cr, x, y0);
    cairo_rotate(cr, ball_roll[i]);
    cairo_scale(cr, 2 * r / 48, 2 * r / 48);
    cairo_set_source_surface(cr, ball_tint[i], -24, -24);
    cairo_paint(cr);
    cairo_restore(cr);
}

/* the strand from each ball up to her needles */
static void draw_strand(cairo_t *cr, int i, double hx, double hy, double tension, double t)
{
    double x = BALL_POS[i][0] + ball_kick[i], y = BALL_POS[i][1] + BALL_RMAX - ball_r[i];
    double ang = atan2(hy - y, hx - x);
    double x0 = x + cos(ang) * ball_r[i] * 0.8, y0 = y + sin(ang) * ball_r[i] * 0.8;
    double sag = 34 * (1 - tension) + 6, wob = 3 * tension * sin(t * 13 + i * 2);
    double mx = (x0 + hx) / 2, my = (y0 + hy) / 2;
    cairo_move_to(cr, x0, y0);
    cairo_curve_to(cr, mx - 20 + wob, my + sag, mx + 10, my + sag * 0.6 + wob, hx, hy);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(cr, 2.8);
    cairo_set_source_rgba(cr, 0.1, 0.04, 0.02, 0.45);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, 1.7);
    set_rgb(cr, ball_col[i]);
    cairo_stroke(cr);
}

static void draw_particles(cairo_t *cr)
{
    for (int i = 0; i < MAX_PARTS; i++) {
        particle *p = &parts[i];
        if (p->life <= 0)
            continue;
        double u = p->age / p->life;
        switch (p->kind) {
        case P_STEAM: {
            double a = (u < 0.2 ? u / 0.2 : 1 - (u - 0.2) / 0.8) * 0.55;
            double d = p->size * (0.5 + 1.3 * u);
            cairo_save(cr);
            cairo_translate(cr, p->x, p->y);
            cairo_scale(cr, d / 32, d / 32);
            cairo_set_source_surface(cr, puff_img, -16, -16);
            cairo_paint_with_alpha(cr, a);
            cairo_restore(cr);
            break;
        }
        case P_NOTE: {
            double a = u < 0.15 ? u / 0.15 : 1 - pow((u - 0.15) / 0.85, 2);
            cairo_save(cr);
            cairo_translate(cr, p->x, p->y);
            cairo_rotate(cr, 0.25 * sin(p->age * 3 + p->spin * 3));
            cairo_scale(cr, p->size, p->size);
            cairo_set_source_surface(cr, note_img[p->sub & 1][p->sub >> 1 & 1], -16, -16);
            cairo_paint_with_alpha(cr, a);
            cairo_restore(cr);
            break;
        }
        case P_ZZZ: {
            double a = u < 0.2 ? u / 0.2 : 1 - (u - 0.2) / 0.8;
            cairo_select_font_face(cr, "DejaVu Serif", CAIRO_FONT_SLANT_ITALIC, CAIRO_FONT_WEIGHT_BOLD);
            cairo_set_font_size(cr, p->size * (0.7 + 0.6 * u));
            cairo_move_to(cr, p->x, p->y);
            cairo_text_path(cr, "z");
            cairo_set_line_width(cr, 3);
            cairo_set_source_rgba(cr, 0.2, 0.1, 0.25, 0.6 * a);
            cairo_stroke_preserve(cr);
            cairo_set_source_rgba(cr, 0.95, 0.92, 1.0, a);
            cairo_fill(cr);
            break;
        }
        case P_GLINT: {
            double a = 1 - u, s = 3 + 3 * (1 - u);
            cairo_move_to(cr, p->x - s, p->y);
            cairo_line_to(cr, p->x + s, p->y);
            cairo_move_to(cr, p->x, p->y - s);
            cairo_line_to(cr, p->x, p->y + s);
            cairo_set_line_width(cr, 1.3);
            cairo_set_source_rgba(cr, 1, 0.95, 0.8, 0.9 * a);
            cairo_stroke(cr);
            break;
        }
        case P_LOOP: {
            /* a dropped stitch: a little loop of yarn tumbling to the floor */
            double a = u < 0.8 ? 1 : 1 - (u - 0.8) / 0.2;
            cairo_save(cr);
            cairo_translate(cr, p->x, p->y);
            cairo_rotate(cr, p->age * 5 * p->spin);
            cairo_scale(cr, 1, 0.6);
            cairo_arc(cr, 0, 0, 5, 0, 2 * M_PI);
            cairo_restore(cr);
            cairo_set_line_width(cr, 2.2);
            cairo_set_source_rgba(cr, ball_col[p->sub].r, ball_col[p->sub].g, ball_col[p->sub].b, a);
            cairo_stroke(cr);
            break;
        }
        }
    }
}

/* ---------------------------------------------------------------- simulation */

static double level(double bytes, double lo, double hi) { return bytes <= lo ? 0 : clamp01(log10(bytes / lo) / log10(hi / lo)); }

static void ease_view(view_t *v, const stats *g, const sys_stats *s, double dt)
{
    double k = fmin(1, dt * 3), ks = fmin(1, dt * 1.2);
    v->rate += (g->tok_s - v->rate) * k;
    for (int i = 0; i < N_GPUS; i++) {
        v->rate_gpu[i] += (g->tok_port[i] - v->rate_gpu[i]) * k;
        v->gtemp[i] += (g->temp[i] - v->gtemp[i]) * ks;
        v->vram[i] += (s->vram[i] - v->vram[i]) * ks;
    }
    v->cpu_load += (s->cpu_load - v->cpu_load) * k;
    v->cpu_temp += (s->cpu_temp - v->cpu_temp) * ks;
    v->ram += ((s->ram_total > 0 ? s->ram_used / s->ram_total : 0) - v->ram) * ks;
    v->disk += (level(s->disk, 1e5, 3e9) - v->disk) * k;
    v->lan += (level(s->net_rx + s->net_tx, 3e3, 1e8) - v->lan) * k;
    v->ts += (level(s->ts_rx + s->ts_tx, 3e3, 3e7) - v->ts) * k;
    v->ctxt += (clamp01((log10(fmax(1, s->ctxt)) - 4.0) / 2.3) - v->ctxt) * ks;
}

static int scene_idle(const stats *g) { return g->tok_s < 1 && g->running == 0; }

static void simulate(const view_t *v, const stats *g, const sys_stats *s, double dt, double t)
{
    int idle = scene_idle(g);

    /* dozing, with hysteresis: nods off after a few quiet seconds, wakes when work arrives */
    idle_for = idle ? idle_for + dt : 0;
    busy_for = idle ? 0 : busy_for + dt;
    if (!dozing && idle_for > DOZE_AFTER)
        dozing = 1;
    if (dozing && busy_for > 0.5)
        dozing = 0;
    doze += ((dozing ? 1.0 : 0.0) - doze) * fmin(1, dt * 1.8);

    /* the rocking chair: gentle, slower and smaller when she sleeps */
    double act = 1 - exp(-v->rate / SCROLL_KNEE);
    rock_amp += ((dozing ? 0.012 : 0.022 + 0.01 * act) - rock_amp) * fmin(1, dt);
    rock_t += dt * (dozing ? 0.55 : 0.8 + 0.3 * act);

    /* knitting: the scarf spills at a speed set by the rate */
    double speed = dozing ? 0 : SCROLL_MAX * act;
    scroll_phase += speed * dt;
    while (scroll_phase >= ROW_H) {
        scroll_phase -= ROW_H;
        push_row(v);
    }
    stitches += v->rate * dt;
    double click_hz = dozing || v->rate < 1 ? 0 : 1.2 + 6.5 * act;
    double prev = needle_phase;
    needle_phase += click_hz * dt;
    if (floor(needle_phase) != floor(prev)) {
        click_flash = 1;
        spawn(P_GLINT, 0, 211 + frand() * 3, 256 + frand() * 3, 0, 0, 0.16, 1);
    }
    click_flash = fmax(0, click_flash - dt * 8);

    /* dropped stitches from bursts of major page faults */
    since_drop += dt;
    drop_acc = dozing ? 0 : drop_acc + s->majflt * dt / MAJF_PER_DROP;
    if (drop_acc >= 1 && since_drop > DROP_GAP) {
        drop_acc = 0;
        since_drop = 0;
        ladder_left = 7;
        ladder_col = 1 + rand() % 3;
        spawn(P_LOOP, stripe_yarn, HAND_AWAKE[0] + 12, HAND_AWAKE[1] + 10, 18, -30, 1.6, 1);
    }
    if (drop_acc > 1)
        drop_acc = 1;

    /* yarn balls: size is free VRAM, they roll as they unwind, the cat bats them about */
    for (int i = 0; i < N_GPUS; i++) {
        double want = BALL_RMIN + (BALL_RMAX - BALL_RMIN) * (1 - clamp01(v->vram[i]));
        ball_r[i] += (want - ball_r[i]) * fmin(1, dt * 1.5);
        double share = v->rate > 1 ? v->rate_gpu[i] / v->rate : 0;
        ball_roll[i] += (speed * share * 0.6 + ball_kick_v[i] * 0.4) * dt / ball_r[i];
        ball_kick_v[i] += (-ball_kick[i] * 30 - ball_kick_v[i] * 4) * dt;
        ball_kick[i] += ball_kick_v[i] * dt;
        ball_kick[i] = fmax(-14, fmin(14, ball_kick[i]));
        rgb c = yarn_color(i, v->gtemp[i]);
        if (fabs(c.r - ball_col[i].r) + fabs(c.g - ball_col[i].g) + fabs(c.b - ball_col[i].b) > 0.02) {
            ball_col[i] = c;
            tint_ball(ball_tint[i], c);
        }
    }

    /* the cat: disks drive her chaos; asleep on the scarf when all is quiet */
    double chaos = clamp01(v->disk * 1.1);
    int want_sleep = dozing && v->disk < 0.06;
    cat_update(chaos, want_sleep, dt);

    /* basket: one ball of yarn per ~11% of RAM in use (with hysteresis) */
    double want_n = v->ram * 9;
    if (fabs(want_n - basket_target) > 0.7)
        basket_target = (int)lround(want_n);
    basket_n = basket_target;
    for (int i = 0; i < 9; i++)
        basket_scale[i] += ((i < basket_n ? 1.0 : 0.0) - basket_scale[i]) * fmin(1, dt * 3);

    /* fire: CPU temperature */
    fire_flick += ((frand() * 2 - 1) - fire_flick) * fmin(1, dt * 8);
    double heat = clamp01((v->cpu_temp - 40) / 50);
    flame_acc += (30 + 80 * heat) * dt;
    while (flame_acc >= 1) {
        flame_acc -= 1;
        for (int i = 0; i < MAX_FLAMES; i++)
            if (flames[i].life <= 0) {
                double spread = 34 + 14 * heat, off = (frand() - 0.5) * spread;
                double centre = 1 - fabs(off) / (spread / 2);     /* taller in the middle */
                flames[i] = (particle){ FIRE_BASE[0] + off, FIRE_BASE[1] + 1, -off * 0.5,
                                        -(18 + 55 * heat) * (0.6 + 0.5 * centre + 0.3 * frand()), 0,
                                        0.35 + 0.35 * frand() + 0.25 * heat * centre, (7 + 9 * heat) * (0.7 + 0.5 * frand()),
                                        0, 0, 0 };
                break;
            }
    }
    for (int i = 0; i < MAX_FLAMES; i++) {
        particle *f = &flames[i];
        if (f->life <= 0)
            continue;
        if ((f->age += dt) >= f->life) {
            f->life = 0;
            continue;
        }
        f->vx += (sin(t * 3 + i) * 10 - f->vx * 0.8) * dt;
        f->x += f->vx * dt;
        f->y += f->vy * dt;
    }

    /* steam from the teapot: CPU load */
    steam_acc += (0.3 + 11 * pow(v->cpu_load, 1.3)) * dt;
    while (steam_acc >= 1) {
        steam_acc -= 1;
        double hard = v->cpu_load;
        spawn(P_STEAM, 0, TEAPOT_X + 44, TEAPOT_Y + 7, 6 + 14 * hard + frand() * 6, -(14 + 26 * hard) - frand() * 8,
              1.6 + frand() * 1.2 + hard, 12 + 10 * hard);
    }
    /* music from the radio: LAN (gold) and tailscale (blue) */
    double lv[2] = { v->lan, v->ts };
    for (int n = 0; n < 2; n++) {
        note_acc[n] += (lv[n] > 0.02 ? 0.3 + 4.5 * pow(lv[n], 1.3) : 0) * dt;
        while (note_acc[n] >= 1) {
            note_acc[n] -= 1;
            spawn(P_NOTE, n | (rand() % 2) << 1, RADIO_X + 12, RADIO_Y + 6, -14 - frand() * 16 - 12 * lv[n],
                  -16 - frand() * 10 - 14 * lv[n], 2.4 + frand(), 0.8 + 0.25 * frand());
        }
    }
    /* zzz */
    if (doze > 0.7) {
        zzz_acc += dt / 1.3;
        if (zzz_acc >= 1) {
            zzz_acc = 0;
            spawn(P_ZZZ, 0, HEAD[0] - 12, HEAD[1] - 6, -24, -13, 3.2, 26);
        }
    }

    for (int i = 0; i < MAX_PARTS; i++) {
        particle *p = &parts[i];
        if (p->life <= 0)
            continue;
        p->age += dt;
        if (p->age >= p->life) {
            p->life = 0;
            continue;
        }
        if (p->kind == P_STEAM) {
            p->vx *= exp(-dt * 0.8);
            p->vx += sin(p->age * 2 + p->spin * 4) * 6 * dt;
        } else if (p->kind == P_NOTE || p->kind == P_ZZZ) {
            p->vx += sin(p->age * 2.5 + p->spin * 5) * 12 * dt;
        } else if (p->kind == P_LOOP) {
            p->vy += 160 * dt;
            if (p->y > 400)
                p->vy = 0, p->vx = 0;
        }
        p->x += p->vx * dt;
        p->y += p->vy * dt;
    }
}

/* ---------------------------------------------------------------- render */

static void fmt_stitches(char *out, size_t cap, double n)
{
    if (n < 100000) {
        long v = (long)n;
        if (v >= 1000)
            snprintf(out, cap, "%ld,%03ld STS", v / 1000, v % 1000);
        else
            snprintf(out, cap, "%ld STS", v);
    } else if (n < 1e6)
        snprintf(out, cap, "%.1fK STS", n / 1e3);
    else if (n < 1e9)
        snprintf(out, cap, "%.*fM STS", n < 1e7 ? 2 : n < 1e8 ? 1 : 0, n / 1e6);
    else
        snprintf(out, cap, "%.2fB STS", n / 1e9);
}

static void update_hud(const view_t *v, const sys_stats *s, double t)
{
    static double shown, next;
    static double shown_st;
    char key[160], a[32], c[32];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        shown = shown_rate(v->rate);
        shown_st = stitches;
    }
    int day = (int)(s->uptime / 86400) + 1;
    int nap = dozing && doze > 0.5;
    snprintf(a, sizeof(a), "%.0f", shown);
    fmt_stitches(c, sizeof(c), shown_st);
    snprintf(key, sizeof(key), "%s|%s|%d|%d|%d", a, c, day, nap, gpu_source);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);

    cairo_t *cr = cairo_create(hud);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    const double cx = (SAMP_X0 + SAMP_X1) / 2;
    const rgb red = { 0.66, 0.10, 0.12 }, green = { 0.22, 0.40, 0.20 }, navy = { 0.18, 0.24, 0.46 };
    const char *top = nap ? "ZZZ" : a, *mid = nap ? "NAPPING" : gpu_source ? "% GPU" : "TOK/S";
    /* the big number fits the cloth: shrink the cell if it has many digits */
    double cell = fmin(4.8, (SAMP_X1 - SAMP_X0 - 24) / (xs_width(top, 1) + 0.01));
    xs_text(cr, top, cx, 51, cell, red);
    xs_text(cr, mid, cx, 90, 3.2, green);
    double cell_c = fmin(3.2, (SAMP_X1 - SAMP_X0 - 20) / xs_width(c, 1));
    xs_text(cr, c, cx, 117, cell_c, navy);

    /* calendar day */
    char d[16];
    snprintf(d, sizeof(d), "%d", day);
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    double fs = 32;
    cairo_set_font_size(cr, fs);
    cairo_text_extents(cr, d, &ext);
    if (ext.width > CAL_W - 10) {
        fs *= (CAL_W - 10) / ext.width;
        cairo_set_font_size(cr, fs);
        cairo_text_extents(cr, d, &ext);
    }
    cairo_move_to(cr, CAL_X + CAL_W / 2 - ext.width / 2 - ext.x_bearing, CAL_Y + 48 - ext.height / 2 - ext.y_bearing);
    cairo_set_source_rgb(cr, 0.22, 0.12, 0.08);
    cairo_show_text(cr, d);
    cairo_destroy(cr);
}

/* The room with the fire's glow on it; rebuilt only when the fire changes size */
static cairo_surface_t *lit;
static int lit_q = -1;

static void update_lit(double heat)
{
    int q = (int)lround(heat * 20);
    int tail_gone = rows_total * ROW_H > path_n + 2 * ROW_H;
    if (q == lit_q && tail_gone == scarf_shadow_baked)
        return;
    scarf_shadow_baked = tail_gone;
    lit_q = q;
    heat = q / 20.0;
    if (!lit)
        lit = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(lit);
    cairo_set_source_surface(cr, bg, 0, 0);
    cairo_paint(cr);
    const double bx = FIRE_BASE[0], by = FIRE_BASE[1] - 10, gr = 130 + 90 * heat;
    cairo_pattern_t *p = cairo_pattern_create_radial(bx, by, 0, bx, by, gr);
    cairo_pattern_add_color_stop_rgba(p, 0, 1.0, 0.55, 0.18, 0.12 + 0.26 * heat);
    cairo_pattern_add_color_stop_rgba(p, 0.45, 1.0, 0.45, 0.12, 0.04 + 0.10 * heat);
    cairo_pattern_add_color_stop_rgba(p, 1, 1.0, 0.4, 0.1, 0);
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_arc(cr, bx, by, gr, 0, 2 * M_PI);
    cairo_set_source(cr, p);
    cairo_fill(cr);
    cairo_pattern_destroy(p);
    if (scarf_shadow_baked) {
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
        cairo_translate(cr, 2, 3.5);
        int i0 = (int)layers[1].s_from + 4;
        for (int i = i0; i < path_n; i += 3)
            (i == i0 ? cairo_move_to : cairo_line_to)(cr, path_x[i], path_y[i]);
        cairo_set_line_width(cr, 28);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
        cairo_set_source_rgba(cr, 0.08, 0.02, 0.0, 0.30);
        cairo_stroke(cr);
    }
    cairo_destroy(cr);
}

/*
 * Grandma rotated about the rockers, cached per angle step (0.0025 rad, about half a
 * pixel at the top of her head) and made only when first needed.
 */
#define GM_STEP         0.0025
#define GM_STEPS        31                      /* -0.0375 .. +0.0375 rad */
#define GM_CX           (GM_X + 28)             /* the cached frames' box on screen */
#define GM_CY           (GM_Y + 4)
#define GM_CW           234
#define GM_CH           274
static cairo_surface_t *gm_rot[2][GM_STEPS];

static double quant_rock(double a)
{
    int k = (int)lround(a / GM_STEP);
    k = k < -(GM_STEPS / 2) ? -(GM_STEPS / 2) : k > GM_STEPS / 2 ? GM_STEPS / 2 : k;
    return k * GM_STEP;
}

static cairo_surface_t *gm_frame(int pose, double a)
{
    int k = (int)lround(a / GM_STEP) + GM_STEPS / 2;
    if (!gm_rot[pose][k]) {
        cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, GM_CW, GM_CH);
        cairo_t *cr = cairo_create(s);
        cairo_translate(cr, -GM_CX, -GM_CY);
        cairo_translate(cr, PIVOT[0], PIVOT[1]);
        cairo_rotate(cr, a);
        cairo_translate(cr, -PIVOT[0], -PIVOT[1]);
        cairo_set_source_surface(cr, gm_img[pose], GM_X, GM_Y);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_GOOD);
        cairo_paint(cr);
        cairo_destroy(cr);
        gm_rot[pose][k] = s;
    }
    return gm_rot[pose][k];
}

static void render(cairo_t *cr, const view_t *v, const sys_stats *s, double t)
{
    double heat = clamp01((v->cpu_temp - 40) / 50);
    update_lit(heat);
    cairo_set_source_surface(cr, lit, 0, 0);
    cairo_paint(cr);

    draw_fire(cr, heat, t);

    /* radio dial glows with network traffic */
    double lv = fmax(v->lan, v->ts);
    if (lv > 0.01) {
        double gx = RADIO_X + 33, gy = RADIO_Y + 13, gr = 10 + 8 * lv;
        cairo_pattern_t *p = cairo_pattern_create_radial(gx, gy, 0, gx, gy, gr);
        cairo_pattern_add_color_stop_rgba(p, 0, 1, 0.8, 0.35, 0.5 * lv);
        cairo_pattern_add_color_stop_rgba(p, 1, 1, 0.7, 0.3, 0);
        cairo_arc(cr, gx, gy, gr, 0, 2 * M_PI);
        cairo_set_source(cr, p);
        cairo_fill(cr);
        cairo_pattern_destroy(p);
    }

    /* teapot lid rattles when the CPU is flat out: redraw it jiggled */
    if (v->cpu_load > 0.8) {
        double j = (v->cpu_load - 0.8) * 5 * sin(t * 47);
        cairo_save(cr);
        cairo_rectangle(cr, TEAPOT_X + 10, TEAPOT_Y - 2, 22, 10);
        cairo_clip(cr);
        cairo_set_source_surface(cr, teapot_img, TEAPOT_X, TEAPOT_Y + j);
        cairo_paint(cr);
        cairo_restore(cr);
    }

    draw_basket(cr, t);

    /* grandma: awake and dozing sprites crossfaded, rocked about the rockers */
    double rock = quant_rock(rock_amp * sin(rock_t * 2 * M_PI * 0.45));
    for (int k = 0; k < 2; k++) {
        double a = k ? doze : 1 - doze;
        if (a < 0.01)
            continue;
        cairo_set_source_surface(cr, gm_frame(k, rock), GM_CX, GM_CY);
        if (a > 0.99)
            cairo_paint(cr);
        else
            cairo_paint_with_alpha(cr, a);
    }
    cairo_save(cr);
    cairo_translate(cr, PIVOT[0], PIVOT[1]);
    cairo_rotate(cr, rock);
    cairo_translate(cr, -PIVOT[0], -PIVOT[1]);
    /* needles click: her hands swing a little, feathered into the sprite */
    if (doze < 0.5 && click_flash > 0) {
        double sw = 0.05 * click_flash * (fmod(needle_phase, 2) < 1 ? 1 : -1);
        double hx = HAND_AWAKE[0] - 6, hy = HAND_AWAKE[1] + 4;
        cairo_save(cr);
        cairo_translate(cr, hx, hy);
        cairo_rotate(cr, sw);
        cairo_translate(cr, -hx, -hy);
        cairo_set_source_surface(cr, gm_img[0], GM_X, GM_Y);
        cairo_restore(cr);
        cairo_mask_surface(cr, hand_mask, hx - 40, hy - 40);
    }
    cairo_restore(cr);

    draw_scarf(cr, rock);

    /* strands and yarn balls */
    double hx0 = HAND_AWAKE[0] + (HAND_DOZE[0] - HAND_AWAKE[0]) * doze;
    double hy0 = HAND_AWAKE[1] + (HAND_DOZE[1] - HAND_AWAKE[1]) * doze;
    double hx, hy;
    rock_point(hx0, hy0, rock, &hx, &hy);
    for (int i = 0; i < N_GPUS; i++) {
        double share = v->rate > 1 ? v->rate_gpu[i] / v->rate : 0;
        double tension = dozing ? 0 : clamp01(share * 1.4) * (1 - exp(-v->rate / SCROLL_KNEE));
        draw_strand(cr, i, hx + 2 * i, hy + 2, tension, t);
    }
    for (int i = 0; i < N_GPUS; i++)
        draw_ball(cr, i, v->gtemp[i]);

    draw_cat(cr, t);
    draw_particles(cr);

    update_hud(v, s, t);
    cairo_rectangle(cr, SAMP_X0, SAMP_Y0, SAMP_X1 - SAMP_X0, SAMP_Y1 - SAMP_Y0);
    cairo_rectangle(cr, CAL_X, CAL_Y + 24, CAL_W, CAL_H - 24);
    cairo_set_source_surface(cr, hud, 0, 0);
    cairo_fill(cr);
}

/* ---------------------------------------------------------------- main */

static void scene_reset(long rows_knitted)
{
    memset(parts, 0, sizeof(parts));
    for (int i = 0; i < MAX_ROWS; i++) {
        rgb c = yarn_color((i / 8) % 2, 50);
        double k = (i / 16) % 2 ? 0.78 : 1.0;
        rows[i] = (row_t){ (float)(c.r * k), (float)(c.g * k), (float)(c.b * k), -1 };
    }
    rows_total = rows_knitted;
    for (int i = 0; i < N_GPUS; i++)
        ball_r[i] = BALL_RMAX, ball_col[i] = (rgb){ -1, -1, -1 };
}

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats g = { 0 };
    sys_stats s = { 0 };
    view_t v = { 0 };
    double last, next_poll = 0, t0;
    int fd = -1, bench = 0, showcase = 0;

    const char *source = getenv("LLM_REACTOR_SOURCE");
    gpu_source = source && !strcasecmp(source, "gpu");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu-load"))
            gpu_source = 1;
        else if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--showcase"))
            showcase = demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    srand((unsigned)time(NULL));
    build_static();
    build_path();
    build_stitch();
    build_scarf();
    cat_init();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* busy / middling / idle, each simulated for a while from the showcase's script */
        struct { double u; const char *png; } scenes[] = {
            { 21.0, "knit_preview.png" }, { 10.0, "knit_mid.png" }, { 35.0, "knit_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(7 + k);
            scene_reset(400);
            memset(&v, 0, sizeof(v));
            hud_key[0] = 0;
            dozing = 0, doze = 0, idle_for = 0;
            cat_init();
            int n = 14 * FPS_BUSY;
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                double t = i / (double)FPS_BUSY;
                showcase_poll(&g, &s, scenes[k].u - 12 + t * 12.0 / 14);
                ease_view(&v, &g, &s, 1.0 / FPS_BUSY);
                if (i == n - 60)
                    b0 = now_s();
                simulate(&v, &g, &s, 1.0 / FPS_BUSY, t);
                render(cr, &v, &s, t);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / 60, len);
            cairo_surface_write_to_png(surf, scenes[k].png);
        }
        return 0;
    }

    scene_reset(showcase ? 70 : 1000);
    dozing = 1, doze = 1;                       /* start asleep; work wakes her */
    if (!demo) {
        gpus_init();
        sys_init();
    }

    while ((fd = lcd_open()) < 0 && !stop)
        sleep(2);
    if (fd >= 0)
        lcd_brightness(fd, 100);

    t0 = last = now_s();
    while (!stop) {
        double t = now_s(), dt = t - last;
        last = t;
        if (dt > 0.25)
            dt = 0.25;

        if (showcase) {
            showcase_poll(&g, &s, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 0.5;
            if (demo) {
                /* scale a copy; scaling the live stats would compound */
                static stats demo_g;
                static sys_stats demo_s;
                demo_poll(&demo_g, &demo_s, t - t0);
                g = demo_g;
                s = demo_s;
            } else {
                gpus_poll(&g);
                sys_poll(&s, t);
                if (gpu_source)
                    gpu_rate_poll(&g);
                else
                    vllm_poll(&g, t);
            }
        }

        ease_view(&v, &g, &s, dt);
        simulate(&v, &g, &s, dt, t - t0);
        render(cr, &v, &s, t - t0);
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

        int quiet = doze > 0.99 && cat.state == CAT_SLEEP && !showcase;
        double spare = 1.0 / (quiet ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
