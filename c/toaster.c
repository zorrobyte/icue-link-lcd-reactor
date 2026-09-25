/*
 * toaster: a retro kitchen on the iCUE LINK AIO pump LCD, run by the whole machine.
 *
 * Two chrome toasters on the counter are the two GPUs (sky blue is GPU 0, tangerine
 * is GPU 1). Their slots glow orange with the card's power draw, and toast pops out
 * and flies onto the plate in front at a rate set by that GPU's tok/s (GPU activity in
 * GPU mode). How done the toast is follows the GPU temperature: pale bread when cool,
 * golden, brown, then burnt black and smoking, and the smoke alarm on the wall goes
 * off at the top temperatures. At peak load a bagel or a waffle launches now and then.
 * Piles topple off the counter when they get too tall.
 *
 * The rest of the kitchen is the rest of the machine: the kettle's gas ring has one
 * flame per CPU thread (/proc/stat), it steams with total CPU load, whistles with CPU
 * temperature (k10temp) and its lid rattles with CPU pressure stalls (PSI). The
 * cookie jar fills with RAM in use. The microwave lights up and turns with NVMe
 * throughput, and popcorn pops inside it with every process the machine forks. The
 * radio plays notes for LAN traffic (enp12s0) and violet notes for tailscale0. The
 * extractor fan spins with the GPU fans, and the electricity meter's disc turns with
 * the total draw (CPU package plus both GPUs). Idle: cold toasters, one slice of
 * bread waiting, and a fly.
 *
 * The kitchen and every object in it are images made with an image model
 * (assets/toaster/). Run with --demo to simulate data, --showcase for a scripted 36 s
 * breakfast rush, --bench to write preview PNGs.
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
    double fan[N_GPUS];         /* fan speed, % */
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
        unsigned int mw, fan;
        if (!nvml_dev[i])
            continue;
        if (nvmlDeviceGetUtilizationRates(nvml_dev[i], &u) == NVML_SUCCESS)
            s->load[i] = u.gpu / 100.0;
        if (nvmlDeviceGetPowerUsage(nvml_dev[i], &mw) == NVML_SUCCESS)
            s->power[i] = mw / 1000.0;
        if (nvmlDeviceGetTemperatureV(nvml_dev[i], &temp) == NVML_SUCCESS)
            s->temp[i] = temp.temperature;
        if (nvmlDeviceGetFanSpeed(nvml_dev[i], &fan) == NVML_SUCCESS)
            s->fan[i] = fan;
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
 * The rest of the kitchen comes from the kernel: per-thread CPU load and forks
 * (/proc/stat), CPU temperature (hwmon "k10temp", found by name, never by number),
 * CPU pressure stalls (/proc/pressure/cpu), RAM (/proc/meminfo), NVMe throughput
 * (/proc/diskstats), network (/proc/net/dev) and CPU package power (RAPL, root only;
 * estimated from load when it can't be read). sys_init() finds the files once;
 * sys_poll() reads them (cheap, meant for ~2 Hz) and differences the counters into
 * rates. Missing sensors just stay at 0.
 */
#define SYS_MAX_THREADS 64
#define N_NVME          3

typedef struct {
    int    n_threads;
    double thread_load[SYS_MAX_THREADS];    /* 0..1 per hardware thread */
    double cpu_load;                        /* 0..1 whole package */
    double cpu_temp;                        /* k10temp Tctl, C */
    double psi_cpu;                         /* % of time some task waited for a CPU (avg10) */
    double forks_s;                         /* new processes per second */
    double ram_total, ram_used;             /* GB (used = total - available) */
    double nvme_rd[N_NVME], nvme_wr[N_NVME];/* bytes/s, nvme0n1..nvme2n1 */
    double net_rx, net_tx;                  /* enp12s0, bytes/s */
    double ts_rx, ts_tx;                    /* tailscale0, bytes/s */
    double pkg_watts;                       /* CPU package power, W */
    int    pkg_measured;                    /* 1: RAPL, 0: estimated from load */
} sys_stats;

static const char *NET_IF = "enp12s0", *TS_IF = "tailscale0";
static const char *RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj";

static char k10_temp_path[300];
static int  rapl_ok = 1;

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
    static unsigned long long last_rd[N_NVME], last_wr[N_NVME], last_net[4], last_forks, last_energy;
    static double last_t;
    static int have;
    double dt = t - last_t;
    char line[8192], b[256];
    FILE *f;

    /* CPU: per-thread busy share since the last poll, and forks */
    if ((f = fopen("/proc/stat", "r"))) {
        unsigned long long all_busy = 0, all_total = 0;
        int n = 0;
        while (fgets(line, sizeof(line), f)) {
            unsigned long long v[8] = { 0 }, x;
            int id;
            if (sscanf(line, "processes %llu", &x) == 1) {
                if (have && dt > 0 && x >= last_forks)
                    s->forks_s = (x - last_forks) / dt;
                last_forks = x;
                continue;
            }
            if (strncmp(line, "cpu", 3) || line[3] == ' ')
                continue;
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
            if (id + 1 > n)
                n = id + 1;
        }
        fclose(f);
        s->n_threads = n;
        if (all_total)
            s->cpu_load = (double)all_busy / all_total;
    }

    /* CPU temperature and pressure stalls */
    if (*k10_temp_path && read_text(k10_temp_path, b, sizeof(b)) > 0)
        s->cpu_temp = strtod(b, NULL) / 1000.0;
    const char *a;
    if (read_text("/proc/pressure/cpu", b, sizeof(b)) > 0 && (a = strstr(b, "some avg10=")))
        s->psi_cpu = strtod(a + 11, NULL);

    /* Package power: RAPL energy counter (root only), else a rough estimate from load */
    if (rapl_ok) {
        if (read_text(RAPL_PATH, b, sizeof(b)) <= 0) {
            rapl_ok = 0;                        /* EACCES or missing: estimate from now on */
        } else {
            unsigned long long e = strtoull(b, NULL, 10);
            if (have && dt > 0 && last_energy && e >= last_energy)
                s->pkg_watts = (e - last_energy) / 1e6 / dt;
            last_energy = e;
            s->pkg_measured = 1;
        }
    }
    if (!rapl_ok) {
        s->pkg_watts = 30 + 170 * s->cpu_load;
        s->pkg_measured = 0;
    }

    /* RAM */
    if ((f = fopen("/proc/meminfo", "r"))) {
        double total = 0, avail = 0, v;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %lf", &v) == 1)
                total = v;
            else if (sscanf(line, "MemAvailable: %lf", &v) == 1) {
                avail = v;
                break;                          /* comes after MemTotal */
            }
        }
        fclose(f);
        s->ram_total = total / 1048576.0;
        s->ram_used = (total - avail) / 1048576.0;
    }

    /* NVMe throughput: sectors read (field 6) and written (field 10), 512 bytes each */
    if ((f = fopen("/proc/diskstats", "r"))) {
        while (fgets(line, sizeof(line), f)) {
            char name[32];
            unsigned long long rd, wr;
            if (sscanf(line, "%*u %*u %31s %*u %*u %llu %*u %*u %*u %llu", name, &rd, &wr) != 3)
                continue;
            int i;
            if (strncmp(name, "nvme", 4) || strlen(name) != 7 || strcmp(name + 5, "n1") ||
                (i = name[4] - '0') < 0 || i >= N_NVME)
                continue;
            if (have && dt > 0) {
                s->nvme_rd[i] = rd >= last_rd[i] ? (rd - last_rd[i]) * 512.0 / dt : 0;
                s->nvme_wr[i] = wr >= last_wr[i] ? (wr - last_wr[i]) * 512.0 / dt : 0;
            }
            last_rd[i] = rd;
            last_wr[i] = wr;
        }
        fclose(f);
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
    last_t = t;
    have = 1;
}

/* ---- end system sensors ---- */

/* ---------------------------------------------------------------- simulated data */

static double frand(void) { return rand() / (double)RAND_MAX; }

/* smooth 0..1 wander made of incommensurate sines */
static double wander(double t, double a, double b)
{
    return clamp01(0.5 + 0.28 * sin(t * a) + 0.22 * sin(t * b + 1.7));
}

/* --demo: everything wanders through idle, busy and flat out, each on its own rhythm */
static void demo_poll(stats *g, sys_stats *s, double t)
{
    double busy = clamp01(0.5 + 0.62 * sin(t * 0.21));
    double b1 = clamp01(busy * 1.1 - 0.1 + 0.15 * sin(t * 0.53));
    double act[2] = { busy, b1 };
    g->tok_s = 0;
    g->running = 0;
    for (int i = 0; i < N_GPUS; i++) {
        double a = act[i] < 0.08 ? 0 : act[i];
        g->load[i] = clamp01(a * 1.1 + (frand() - 0.5) * 0.05);
        g->power[i] = 30 + 520 * a;
        g->temp[i] = (int)(34 + 50 * clamp01(0.5 * a + 0.5 * wander(t, 0.09 + i * 0.03, 0.23)));
        g->fan[i] = 30 + 65 * a;
        g->tok_port[i] = a * GPU_FULL_RATE * (gpu_source ? 1 : 0.6);
        g->running_port[i] = a > 0 ? 1 + (int)(a * 4) : 0;
        g->tok_s += g->tok_port[i];
        g->running += g->running_port[i];
    }

    double cpu = wander(t, 0.17, 0.41);
    s->n_threads = 32;
    s->cpu_load = 0;
    for (int i = 0; i < 32; i++) {
        double l = clamp01(cpu * (0.6 + 0.8 * wander(t + i * 3.1, 0.7 + i * 0.013, 1.9)) - 0.08);
        s->thread_load[i] = l;
        s->cpu_load += l / 32;
    }
    s->cpu_temp = 42 + 48 * clamp01(0.3 * cpu + 0.7 * wander(t, 0.11, 0.07));
    s->psi_cpu = 25 * clamp01((cpu - 0.6) / 0.4) * wander(t, 1.3, 0.5);
    s->forks_s = 400 * pow(wander(t, 0.31, 0.83), 3);
    s->ram_total = 91.9;
    s->ram_used = 12 + 72 * wander(t, 0.05, 0.13);
    double disk = pow(wander(t, 0.27, 0.61), 4);
    for (int i = 0; i < N_NVME; i++) {
        s->nvme_rd[i] = i == 0 ? disk * 5e9 : disk * 4e8;
        s->nvme_wr[i] = i == 0 ? disk * 1.5e9 * wander(t, 0.9, 0.3) : 0;
    }
    double net = pow(wander(t, 0.19, 0.47), 3);
    s->net_rx = net * 8e7;
    s->net_tx = net * 1e7;
    s->ts_rx = pow(wander(t, 0.23, 0.71), 5) * 2e7;
    s->ts_tx = s->ts_rx * 0.3;
    s->pkg_watts = 30 + 170 * s->cpu_load;
    s->pkg_measured = 0;
}

/*
 * --showcase: a scripted 36 s breakfast rush for filming or GIFs. A quiet kitchen with
 * a fly; GPU 0 warms up and pops pale toast; GPU 1 joins while the CPU puts the kettle
 * on, the disks run the microwave and a build forks popcorn; everything flat out and
 * too hot: burnt toast, bagels and waffles, smoke alarm, whistling kettle, piles
 * toppling; then it all cools down to one slice of bread and the fly again.
 */
static double ramp(double u, double a, double b) { return clamp01((u - a) / (b - a)); }

static void showcase_poll(stats *g, sys_stats *s, double t)
{
    double u = fmod(t, 36.0);
    double a0 = 0, a1 = 0, heat0, heat1, cpu, hot;

    a0 = 0.55 * ramp(u, 4, 6) + 0.45 * ramp(u, 11, 13) - 0.7 * ramp(u, 26, 28) - 0.3 * ramp(u, 30, 31.5);
    a1 = 0.75 * ramp(u, 10, 12) + 0.25 * ramp(u, 16, 17) - 0.6 * ramp(u, 26.5, 28.5) - 0.4 * ramp(u, 30, 31.5);
    a0 = clamp01(a0);
    a1 = clamp01(a1);
    hot = ramp(u, 16, 21) - ramp(u, 25, 29);                     /* the burnt stretch */
    heat0 = clamp01(0.08 + 0.35 * ramp(u, 5, 11) + 0.25 * ramp(u, 11, 16) + 0.34 * hot - 0.62 * ramp(u, 28, 33));
    heat1 = clamp01(0.10 + 0.30 * ramp(u, 10, 14) + 0.22 * ramp(u, 14, 17) + 0.38 * hot - 0.60 * ramp(u, 28, 33));
    double act[2] = { a0, a1 }, heat[2] = { heat0, heat1 };
    g->tok_s = 0;
    g->running = 0;
    for (int i = 0; i < N_GPUS; i++) {
        double a = act[i] + (act[i] > 0.05 ? (frand() - 0.5) * 0.04 : 0);
        a = clamp01(a);
        g->load[i] = a;
        g->power[i] = 28 + 530 * a;
        g->temp[i] = (int)(32 + 52 * heat[i]);
        g->fan[i] = 30 + 70 * clamp01(0.4 * a + 0.6 * heat[i]);
        g->tok_port[i] = a < 0.03 ? 0 : a * GPU_FULL_RATE * (gpu_source ? 1 : 0.9);
        g->running_port[i] = a > 0.03 ? 1 + (int)(a * 4) : 0;
        g->tok_s += g->tok_port[i];
        g->running += g->running_port[i];
    }

    cpu = clamp01(0.04 + 0.3 * ramp(u, 6, 9) + 0.6 * ramp(u, 14, 18) - 0.55 * ramp(u, 25, 28) - 0.35 * ramp(u, 30, 32));
    s->n_threads = 32;
    s->cpu_load = 0;
    for (int i = 0; i < 32; i++) {
        double l = clamp01(cpu * (0.55 + 0.9 * wander(t * 2 + i * 1.7, 0.9 + i * 0.05, 2.3)) - 0.03);
        s->thread_load[i] = l;
        s->cpu_load += l / 32;
    }
    s->cpu_temp = 45 + 45 * clamp01(0.2 * ramp(u, 6, 10) + 0.8 * ramp(u, 15, 20) - ramp(u, 25, 30));
    s->psi_cpu = 22 * (ramp(u, 18, 19) - ramp(u, 24, 25));
    s->forks_s = 900 * (ramp(u, 11, 11.5) - ramp(u, 16, 16.5)) + 150 * (ramp(u, 19, 20) - ramp(u, 23, 24)) + 3;
    s->ram_total = 91.9;
    s->ram_used = 14 + 58 * (ramp(u, 8, 20) - ramp(u, 27, 34));
    double disk = ramp(u, 7, 7.5) - ramp(u, 10.5, 11) + 0.6 * (ramp(u, 20, 20.5) - ramp(u, 24, 24.5));
    for (int i = 0; i < N_NVME; i++) {
        s->nvme_rd[i] = i == 0 ? disk * 6e9 : 0;
        s->nvme_wr[i] = i == 0 ? disk * 1e9 : 0;
    }
    double net = 0.02 + 0.5 * ramp(u, 5, 6) + 0.48 * (ramp(u, 17, 18) - ramp(u, 25, 26)) - 0.45 * ramp(u, 29, 30);
    s->net_rx = net * 9e7;
    s->net_tx = net * 1e7;
    s->ts_rx = 3e7 * (ramp(u, 13, 14) - ramp(u, 24, 25)) + 2e3;
    s->ts_tx = s->ts_rx * 0.3;
    s->pkg_watts = 30 + 170 * s->cpu_load;
    s->pkg_measured = 0;
}

/* ---------------------------------------------------------------- kitchen */

#define FPS_BUSY        24
#define FPS_IDLE        15
#define GRAVITY         900.0       /* px/s^2 */
#define MAX_POPS_PER_S  1.8         /* per toaster at GPU_FULL_RATE; two slices per pop */
#define PILE_MAX        20          /* slices on a plate before the pile topples */
#define PILE_STEP       4.2         /* px each slice adds to a pile */
#define MAX_FLYERS      160
#define MAX_PARTS       360
#define MAX_CORN        48
#define N_LVL           6           /* doneness: bread, light, golden, brown, dark, burnt */
#define ALARM_ON_C      80          /* hottest GPU at or above this sets the smoke alarm off */
#define ALARM_OFF_C     76

/* Layout: where every object sits on the 480x480 screen */
#define SHELF_X         25
#define SHELF_Y         196         /* shelf sprite top; things on it stand at SHELF_Y + 6 */
#define ON_SHELF        (SHELF_Y + 6)
#define MW_X            30          /* microwave */
#define MW_Y            (ON_SHELF - 59)
#define TIMER_X         142
#define TIMER_Y         (ON_SHELF - 94)
#define LCD_X           (TIMER_X + 20)
#define LCD_Y           (TIMER_Y + 20)
#define LCD_W           158
#define LCD_H           42
#define RADIO_X         338
#define RADIO_Y         (ON_SHELF - 83)
#define FAN_X           72
#define FAN_Y           52
#define METER_X         335
#define METER_Y         55
#define ALARM_X         211
#define ALARM_Y         19
#define HOB_X           202
#define HOB_Y           264
#define KETTLE_X        198
#define KETTLE_Y        186
#define JAR_X           213
#define JAR_Y           299
#define PLATE_Y         331
#define PILE_BASE       357         /* centre of a pile's first slice */

static const double toaster_x[2] = { 42, 298 }, toaster_y[2] = { 232, 240 };
static const double plate_cx[2] = { 112, 368 };
/* the two slices each toaster pops: x centre, the y they rise out from, slice width */
static const double slot_x[2][2] = { { 86, 127 }, { 350, 386 } };
static const double slot_clip[2][2] = { { 250, 250 }, { 247, 257 } };
static const double slot_w[2] = { 28, 32 };
/* slot openings, for the glow: x, y, w, h (relative to the toaster) */
static const double slot_rect[2][2][4] = {
    { { 31, 4, 27, 22 }, { 71, 4, 29, 22 } },
    { { 33, 2, 71, 7 }, { 31, 10, 73, 10 } },
};

enum { K_TOAST, K_BAGEL, K_WAFFLE };
enum { P_RISE, P_FLY, P_FALL };

typedef struct {
    int    alive, kind, lvl, gpu, phase, miss;
    double x, y, vx, vy, ang, spin, scale, clip, t, T, tx, ty, smoke_acc;
} flyer;

typedef struct { double dx, ang, scale; int kind, lvl; } pile_slice;

typedef struct {
    pile_slice s[PILE_MAX];
    int    n, dirty;
    double smoke;               /* seconds of smoke left from a burnt slice on top */
    cairo_surface_t *cache;
} pile_t;

enum { PT_STEAM, PT_SMOKE, PT_NOTE_LAN, PT_NOTE_TS };

typedef struct { int alive, kind, glyph; double x, y, vx, vy, age, life, s0, s1, a0, seed; } part;
typedef struct { int alive, rest; double x, y, vx, vy, ang; } kernel;

typedef struct {
    double tok, rate[2], glow[2], done[2];
    double cpu, thread[SYS_MAX_THREADS], whistle, psi, ram, disk, net, ts, fan, watts, cpu_temp;
} shown_t;

static cairo_surface_t *kitchen_img, *static_layer, *fan_img, *toaster_img[2], *kettle_img, *jar_img, *plate_img;
static cairo_surface_t *toast_img[N_LVL], *bagel_img[N_LVL], *waffle_img[N_LVL], *cookie_img;
#define N_PUFF          32          /* pre-rendered puff radii 1..N_PUFF px */
static cairo_surface_t *puff_spr[2][N_PUFF + 1];  /* [steam, smoke][radius] */
static cairo_surface_t *note_img[2][2], *beep_img, *lcd_cache, *text_cache, *cookie_cache;
static char lcd_key[64], text_key[96];
static int cookie_n = -1;

static flyer   flyers[MAX_FLYERS];
static pile_t  piles[2];
static part    parts[MAX_PARTS];
static kernel  corn[MAX_CORN];
static double  pop_acc[2], last_pop[2] = { -9, -9 }, idle_for, ding_t = -9, steam_acc, lid_acc, note_acc[2], corn_acc;
static double  fan_ang, meter_phase, table_ang, needle, alarm_t;
static int     lvl_cur[2] = { 1, 1 }, alarm_on, gpu_idle;
static double  bread_a;         /* the lone slice waiting on the plate when idle */
static double  fly_x = 520, fly_y = 300, fly_vx, fly_vy, fly_tx = 200, fly_ty = 300, fly_retarget;

/* ---------------------------------------------------------------- assets */

/* Assets live next to the binary (assets/toaster/), or in ./assets/toaster when run from the repo */
static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/toaster/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/toaster/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "toaster: can't load %s (looked next to the binary and in ./assets/toaster)\n", name);
        exit(1);
    }
    return s;
}

/* A writable ARGB copy (PNGs may load as RGB24) */
static cairo_surface_t *argb_copy(cairo_surface_t *src)     /* takes ownership of src */
{
    int w = cairo_image_surface_get_width(src), h = cairo_image_surface_get_height(src);
    cairo_surface_t *d = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *c = cairo_create(d);
    cairo_set_source_surface(c, src, 0, 0);
    cairo_paint(c);
    cairo_destroy(c);
    cairo_surface_flush(d);
    cairo_surface_destroy(src);
    return d;
}

/*
 * Doneness from one lightly toasted slice: pale bread is the slice washed toward
 * cream, the darker levels multiply it down toward charcoal, with char blotches on the
 * burnt one. Every level keeps the same shape and crumb texture.
 */
static cairo_surface_t *toast_level(cairo_surface_t *base, int lvl)
{
    static const double mul[N_LVL][3] = {
        { 1, 1, 1 }, { 1.04, 1.02, 1.0 }, { 1, 1, 1 }, { 0.80, 0.60, 0.42 }, { 0.56, 0.38, 0.24 }, { 0.30, 0.22, 0.17 },
    };
    cairo_surface_t *d = argb_copy(cairo_surface_reference(base));
    int w = cairo_image_surface_get_width(d), h = cairo_image_surface_get_height(d);
    int stride = cairo_image_surface_get_stride(d);
    unsigned char *px = cairo_image_surface_get_data(d);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint32_t *p = (uint32_t *)(px + y * stride) + x;
            double a = (*p >> 24) / 255.0;
            if (a <= 0)
                continue;
            double r = ((*p >> 16) & 255) / 255.0 / a, g = ((*p >> 8) & 255) / 255.0 / a, b = (*p & 255) / 255.0 / a;
            double L = 0.3 * r + 0.59 * g + 0.11 * b;
            if (lvl <= 1) {
                /* bread: cream crumb that keeps the texture, crust stays a little golden */
                double k = lvl == 0 ? 0.78 : 0.4;
                double f = 0.62 + 0.5 * L;
                r = r + (0.99 * f - r) * k;
                g = g + (0.93 * f - g) * k;
                b = b + (0.78 * f - b) * k;
            }
            r *= mul[lvl][0];
            g *= mul[lvl][1];
            b *= mul[lvl][2];
            if (lvl == 5) {
                unsigned hsh = (unsigned)((x / 3) * 73856093u ^ (y / 3) * 19349663u);
                hsh ^= hsh >> 13;
                hsh *= 0x5bd1e995u;
                double blot = 0.55 + 0.45 * ((hsh >> 8) & 255) / 255.0;
                r *= blot, g *= blot, b *= blot;
            }
            r = clamp01(r) * a, g = clamp01(g) * a, b = clamp01(b) * a;
            *p = (*p & 0xff000000u) | ((uint32_t)(r * 255 + 0.5) << 16) | ((uint32_t)(g * 255 + 0.5) << 8) |
                 (uint32_t)(b * 255 + 0.5);
        }
    cairo_surface_mark_dirty(d);
    return d;
}

static void blit(cairo_t *cr, cairo_surface_t *img, double x, double y)
{
    cairo_set_source_surface(cr, img, x, y);
    cairo_paint(cr);
}

static cairo_surface_t *text_sprite(const char *s, double size, rgb fill, rgb line, double lw)
{
    cairo_surface_t *tmp = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 4, 4);
    cairo_t *c = cairo_create(tmp);
    cairo_text_extents_t e;
    cairo_select_font_face(c, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, size);
    cairo_text_extents(c, s, &e);
    cairo_destroy(c);
    cairo_surface_destroy(tmp);
    int w = (int)ceil(e.width + 2 * lw + 4), h = (int)ceil(e.height + 2 * lw + 4);
    cairo_surface_t *img = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    c = cairo_create(img);
    cairo_select_font_face(c, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, size);
    cairo_move_to(c, lw + 2 - e.x_bearing, lw + 2 - e.y_bearing);
    cairo_text_path(c, s);
    cairo_set_line_join(c, CAIRO_LINE_JOIN_ROUND);
    if (lw > 0) {
        cairo_set_line_width(c, lw * 2);
        set_rgb(c, line);
        cairo_stroke_preserve(c);
    }
    set_rgb(c, fill);
    cairo_fill(c);
    cairo_destroy(c);
    return img;
}

static void load_assets(void)
{
    kitchen_img = load_asset("kitchen.png");
    fan_img = argb_copy(load_asset("fan.png"));
    toaster_img[0] = load_asset("toaster_blue.png");
    toaster_img[1] = load_asset("toaster_orange.png");
    kettle_img = load_asset("kettle.png");
    jar_img = load_asset("jar.png");
    plate_img = load_asset("plate.png");
    cookie_img = load_asset("cookie.png");
    cairo_surface_t *toast = load_asset("toast.png"), *bagel = load_asset("bagel.png"), *waffle = load_asset("waffle.png");
    for (int l = 0; l < N_LVL; l++) {
        toast_img[l] = toast_level(toast, l);
        bagel_img[l] = toast_level(bagel, l < 2 ? 2 : l);     /* a bagel is never raw dough */
        waffle_img[l] = toast_level(waffle, l < 2 ? 2 : l);
    }

    /* The extractor fan: punch out the dark opening behind the grille so blades show */
    {
        int w = cairo_image_surface_get_width(fan_img), h = cairo_image_surface_get_height(fan_img);
        int stride = cairo_image_surface_get_stride(fan_img);
        unsigned char *px = cairo_image_surface_get_data(fan_img);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                uint32_t *p = (uint32_t *)(px + y * stride) + x;
                double a = (*p >> 24) / 255.0, d = hypot(x - 36.0, y - 34.5);
                if (a <= 0 || d > 25.5)
                    continue;
                double L = (0.3 * ((*p >> 16) & 255) + 0.59 * ((*p >> 8) & 255) + 0.11 * (*p & 255)) / 255.0 / a;
                double keep = clamp01((L - 0.18) / 0.22);
                double k = keep + (1 - keep) * clamp01((d - 24) / 1.5);
                uint32_t A = (uint32_t)((*p >> 24) * k), R = (uint32_t)(((*p >> 16) & 255) * k),
                         G = (uint32_t)(((*p >> 8) & 255) * k), B = (uint32_t)((*p & 255) * k);
                *p = A << 24 | R << 16 | G << 8 | B;
            }
        cairo_surface_mark_dirty(fan_img);
    }

    /* Static layer: the kitchen and everything that never moves */
    static_layer = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *c = cairo_create(static_layer);
    blit(c, kitchen_img, 0, 0);
    /* dark round rim so the circle's edge feels intentional */
    cairo_pattern_t *vg = cairo_pattern_create_radial(240, 240, 190, 240, 240, 242);
    cairo_pattern_add_color_stop_rgba(vg, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(vg, 1, 0.08, 0.04, 0.02, 0.45);
    cairo_set_source(c, vg);
    cairo_paint(c);
    cairo_pattern_destroy(vg);
    cairo_surface_t *meter = load_asset("meter.png"), *shelf = load_asset("shelf.png"), *mw = load_asset("microwave.png");
    cairo_surface_t *timer = load_asset("timer.png"), *radio = argb_copy(load_asset("radio.png"));
    cairo_surface_t *alarm = load_asset("alarm.png"), *hob = load_asset("hob.png");
    /* the radio's needle moves: paint over the printed one with the dial beside it */
    {
        int stride = cairo_image_surface_get_stride(radio);
        unsigned char *px = cairo_image_surface_get_data(radio);
        for (int y = 36; y <= 53; y++)
            for (int x = 72; x <= 78; x++)
                ((uint32_t *)(px + y * stride))[x] = ((uint32_t *)(px + y * stride))[x - 8];
        cairo_surface_mark_dirty(radio);
    }
    /* soft contact shadows under things on the shelf and counter */
    struct { double x, y, w; } sh[] = {
        { MW_X + 56, ON_SHELF - 1, 118 }, { TIMER_X + 98, ON_SHELF - 1, 190 }, { RADIO_X + 56, ON_SHELF - 1, 110 },
        { toaster_x[0] + 70, 345, 150 }, { toaster_x[1] + 70, 345, 150 }, { 240, 333, 84 },
        { plate_cx[0], 382, 120 }, { plate_cx[1], 382, 120 }, { 240, 381, 58 },
    };
    blit(c, shelf, SHELF_X, SHELF_Y);
    for (unsigned i = 0; i < sizeof(sh) / sizeof(sh[0]); i++) {
        cairo_save(c);
        cairo_translate(c, sh[i].x, sh[i].y);
        cairo_scale(c, sh[i].w / 2, 5);
        cairo_pattern_t *p = cairo_pattern_create_radial(0, 0, 0, 0, 0, 1);
        cairo_pattern_add_color_stop_rgba(p, 0, 0.18, 0.08, 0.02, 0.45);
        cairo_pattern_add_color_stop_rgba(p, 1, 0.18, 0.08, 0.02, 0);
        cairo_set_source(c, p);
        cairo_arc(c, 0, 0, 1, 0, 2 * M_PI);
        cairo_fill(c);
        cairo_pattern_destroy(p);
        cairo_restore(c);
    }
    blit(c, meter, METER_X, METER_Y);
    blit(c, mw, MW_X, MW_Y);
    blit(c, radio, RADIO_X, RADIO_Y);
    blit(c, timer, TIMER_X, TIMER_Y);
    blit(c, alarm, ALARM_X, ALARM_Y);
    blit(c, hob, HOB_X, HOB_Y);
    cairo_destroy(c);
    cairo_surface_destroy(meter);
    cairo_surface_destroy(shelf);
    cairo_surface_destroy(mw);
    cairo_surface_destroy(timer);
    cairo_surface_destroy(radio);
    cairo_surface_destroy(alarm);
    cairo_surface_destroy(hob);
    cairo_surface_destroy(toast);
    cairo_surface_destroy(bagel);
    cairo_surface_destroy(waffle);

    /* Soft round puffs for steam (white) and smoke (charcoal), one per radius, so
       drawing one is a plain unscaled blit */
    for (int k = 0; k < 2; k++)
        for (int r = 1; r <= N_PUFF; r++) {
            puff_spr[k][r] = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 2 * r, 2 * r);
            c = cairo_create(puff_spr[k][r]);
            cairo_pattern_t *p = cairo_pattern_create_radial(r, r, 0, r, r, r);
            double v = k ? 0.34 : 1.0;
            cairo_pattern_add_color_stop_rgba(p, 0, v, v * 0.95, v * 0.92, 1);
            cairo_pattern_add_color_stop_rgba(p, 0.45, v, v * 0.95, v * 0.92, 0.55);
            cairo_pattern_add_color_stop_rgba(p, 1, v, v * 0.95, v * 0.92, 0);
            cairo_set_source(c, p);
            cairo_paint(c);
            cairo_pattern_destroy(p);
            cairo_destroy(c);
        }

    /* Music notes: LAN in warm gold, tailscale in violet */
    static const char *glyph[2] = { "♪", "♫" };
    rgb col[2] = { { 1.0, 0.80, 0.30 }, { 0.72, 0.52, 1.0 } };
    for (int k = 0; k < 2; k++)
        for (int g = 0; g < 2; g++)
            note_img[k][g] = text_sprite(glyph[g], 22, col[k], (rgb){ 0.25, 0.12, 0.05 }, 1.6);
    beep_img = text_sprite("BEEP!", 26, (rgb){ 0.92, 0.10, 0.08 }, (rgb){ 1, 1, 1 }, 3);

    lcd_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, LCD_W, LCD_H);
    text_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 320, 64);
    cookie_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 54, 83);
    for (int g = 0; g < 2; g++) {
        piles[g].cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 180, 160);
        piles[g].dirty = 1;
    }
}

/* ---------------------------------------------------------------- simulation */

static void add_part(int kind, double x, double y, double vx, double vy, double life, double s0, double s1, double a0)
{
    for (int i = 0; i < MAX_PARTS; i++)
        if (!parts[i].alive) {
            parts[i] = (part){ 1, kind, rand() & 1, x, y, vx, vy, 0, life, s0, s1, a0, frand() * 6 };
            return;
        }
}

static flyer *new_flyer(void)
{
    for (int i = 0; i < MAX_FLYERS; i++)
        if (!flyers[i].alive) {
            memset(&flyers[i], 0, sizeof(flyers[i]));
            flyers[i].alive = 1;
            return &flyers[i];
        }
    return NULL;
}

static double item_h(const flyer *f)
{
    cairo_surface_t *img = f->kind == K_BAGEL ? bagel_img[0] : f->kind == K_WAFFLE ? waffle_img[0] : toast_img[0];
    return cairo_image_surface_get_height(img) * f->scale;
}

/* Pop: two slices (or, at peak load, now and then one bagel or waffle) shoot up out of the slots */
static void pop_toaster(int g, int peak, double t)
{
    int special = peak && frand() < 0.16 ? (frand() < 0.5 ? K_BAGEL : K_WAFFLE) : K_TOAST;
    for (int k = 0; k < (special ? 1 : 2); k++) {
        flyer *f = new_flyer();
        if (!f)
            return;
        f->kind = special;
        f->gpu = g;
        f->lvl = lvl_cur[g];
        f->phase = P_RISE;
        f->scale = special ? 1.0 : slot_w[g] / 38.0;
        f->x = special ? (slot_x[g][0] + slot_x[g][1]) / 2 : slot_x[g][k];
        f->clip = special ? fmax(slot_clip[g][0], slot_clip[g][1]) : slot_clip[g][k];
        f->y = f->clip + item_h(f) / 2 - 3;
        f->vy = -(special ? 470 : 345 + 65 * frand()) - 20 * k;
        f->miss = !special && peak && frand() < 0.12;
    }
    last_pop[g] = t;
}

static void land(flyer *f)
{
    pile_t *p = &piles[f->gpu];
    if (p->n < PILE_MAX) {
        pile_slice *s = &p->s[p->n++];
        s->dx = f->x - plate_cx[f->gpu];
        s->ang = (frand() - 0.5) * 0.9;
        s->scale = f->scale;
        s->kind = f->kind;
        s->lvl = f->lvl;
        p->dirty = 1;
        if (f->lvl == 5)
            p->smoke = 5;
    }
    f->alive = 0;
}

/* Too tall (or left alone): the whole pile tips off the counter */
static void topple(int g)
{
    pile_t *p = &piles[g];
    double dir = g == 0 ? -1 : 1;
    for (int i = 0; i < p->n; i++) {
        flyer *f = new_flyer();
        if (!f)
            break;
        f->kind = p->s[i].kind;
        f->lvl = p->s[i].lvl;
        f->gpu = g;
        f->scale = p->s[i].scale;
        f->phase = P_FALL;
        f->x = plate_cx[g] + p->s[i].dx;
        f->y = PILE_BASE - i * PILE_STEP;
        f->vx = dir * (40 + 160 * frand()) + (frand() - 0.5) * 80;
        f->vy = -(60 + 5 * i + 120 * frand());
        f->ang = p->s[i].ang;
        f->spin = (frand() - 0.5) * 10;
    }
    p->n = 0;
    p->dirty = 1;
    p->smoke = 0;
}

static void simulate(const stats *s, const sys_stats *y, const shown_t *sh, double dt, double t)
{
    gpu_idle = s->tok_s < 1 && s->running == 0;
    idle_for = gpu_idle ? idle_for + dt : 0;

    /* Doneness levels with hysteresis, so the colour never flickers between two */
    for (int g = 0; g < 2; g++) {
        double want = sh->done[g] * (N_LVL - 1);
        if (want > lvl_cur[g] + 0.65 && lvl_cur[g] < N_LVL - 1)
            lvl_cur[g]++;
        else if (want < lvl_cur[g] - 0.65 && lvl_cur[g] > 0)
            lvl_cur[g]--;
    }
    int hottest = s->temp[0] > s->temp[1] ? s->temp[0] : s->temp[1];
    if (!alarm_on && hottest >= ALARM_ON_C)
        alarm_on = 1, alarm_t = t;
    else if (alarm_on && hottest < ALARM_OFF_C)
        alarm_on = 0;

    /* Toasters: pops per second follow the square root of each GPU's rate */
    for (int g = 0; g < 2; g++) {
        double r = clamp01(s->tok_port[g] / GPU_FULL_RATE);
        if (s->tok_port[g] < 1) {
            pop_acc[g] = 0.6;              /* first pop comes quickly once work starts */
            continue;
        }
        pop_acc[g] += MAX_POPS_PER_S * sqrt(r) * dt;
        if (pop_acc[g] >= 1) {
            pop_acc[g] -= 1 + (frand() - 0.5) * 0.3;
            pop_toaster(g, r > 0.85, t);
        }
    }

    /* Flying toast */
    for (int i = 0; i < MAX_FLYERS; i++) {
        flyer *f = &flyers[i];
        if (!f->alive)
            continue;
        f->t += dt;
        f->vy += GRAVITY * dt;
        f->x += f->vx * dt;
        f->y += f->vy * dt;
        if (f->phase == P_RISE) {
            if (f->y + item_h(f) / 2 < f->clip) {
                /* clear of the slot: aim for the pile (or, now and then, for the floor) */
                pile_t *p = &piles[f->gpu];
                f->phase = P_FLY;
                f->tx = f->miss ? f->x + (f->gpu ? 1 : -1) * (200 + 120 * frand()) : plate_cx[f->gpu] + (frand() - 0.5) * 34;
                f->ty = f->miss ? 560 : PILE_BASE - (p->n + 0.5) * PILE_STEP;
                double dy = f->ty - f->y;
                f->T = (-f->vy + sqrt(f->vy * f->vy + 2 * GRAVITY * fmax(dy, 1))) / GRAVITY;
                f->vx = (f->tx - f->x) / f->T;
                f->spin = (frand() < 0.5 ? -1 : 1) * (3 + 5 * frand()) * (f->kind == K_TOAST ? 1 : 0.6);
                f->t = 0;
            }
        } else {
            f->ang += f->spin * dt;
            if (f->phase == P_FLY && !f->miss && f->t >= f->T) {
                f->x = f->tx;
                land(f);
                continue;
            }
            if (f->y > 540 || f->x < -60 || f->x > SIZE + 60)
                f->alive = 0;
        }
        /* burnt things trail smoke */
        if (f->lvl >= 4 && f->phase != P_RISE) {
            f->smoke_acc += dt * (f->lvl == 5 ? 12 : 5);
            while (f->smoke_acc >= 1) {
                f->smoke_acc -= 1;
                add_part(PT_SMOKE, f->x + (frand() - 0.5) * 10, f->y - 4, (frand() - 0.5) * 20, -30 - 20 * frand(), 1.1,
                         5, 15, f->lvl == 5 ? 0.45 : 0.25);
            }
        }
    }
    for (int g = 0; g < 2; g++) {
        if (piles[g].n >= PILE_MAX || (idle_for > 2.5 && piles[g].n > 0))
            topple(g);
        if (piles[g].smoke > 0) {
            piles[g].smoke -= dt;
            if (frand() < dt * 10)
                add_part(PT_SMOKE, plate_cx[g] + (frand() - 0.5) * 30, PILE_BASE - piles[g].n * PILE_STEP - 4,
                         (frand() - 0.5) * 10, -25, 1.4, 6, 22, 0.4);
        }
        /* hot slots smoke when the toast is dark */
        if (lvl_cur[g] >= 4 && sh->glow[g] > 0.3 && frand() < dt * (lvl_cur[g] == 5 ? 14 : 4))
            add_part(PT_SMOKE, slot_x[g][rand() & 1] + (frand() - 0.5) * 16, toaster_y[g] + 2, (frand() - 0.5) * 12,
                     -35 - 15 * frand(), 1.8, 7, 30, lvl_cur[g] == 5 ? 0.6 : 0.3);
    }

    /* Kettle: steam with CPU load, a jet when it whistles, lid puffs with pressure stalls */
    steam_acc += dt * (1.2 + 14 * sh->cpu + 18 * sh->whistle);
    while (steam_acc >= 1) {
        steam_acc -= 1;
        double jet = sh->whistle;
        add_part(PT_STEAM, KETTLE_X + 4, KETTLE_Y + 22, -18 - 60 * jet - 12 * frand(), -45 - 20 * frand() + 20 * jet,
                 1.0 + 0.4 * sh->cpu, 3, 10 + 10 * sh->cpu, 0.22 + 0.25 * sh->cpu);
    }
    lid_acc += dt * 8 * clamp01(sh->psi / 15);
    while (lid_acc >= 1) {
        lid_acc -= 1;
        add_part(PT_STEAM, KETTLE_X + 30 + frand() * 25, KETTLE_Y + 30, (frand() - 0.5) * 20, -35, 0.9, 4, 12, 0.35);
    }

    /* Radio: notes for LAN traffic (gold) and tailscale (violet) */
    double nrate[2] = { 0.3 + 3.2 * sh->net, 2.6 * sh->ts };
    for (int k = 0; k < 2; k++) {
        if ((k == 0 && sh->net < 0.02) || (k == 1 && sh->ts < 0.02))
            continue;
        note_acc[k] += dt * nrate[k];
        while (note_acc[k] >= 1) {
            note_acc[k] -= 1;
            add_part(PT_NOTE_LAN + k, RADIO_X + 26 + (frand() - 0.5) * 24, RADIO_Y + 34, -12 + 40 * frand(),
                     -22 - 10 * frand(), 1.6, 1, 1, 1);
        }
    }

    for (int i = 0; i < MAX_PARTS; i++) {
        part *p = &parts[i];
        if (!p->alive)
            continue;
        p->age += dt;
        if (p->age >= p->life) {
            p->alive = 0;
            continue;
        }
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        if (p->kind == PT_STEAM || p->kind == PT_SMOKE) {
            p->vx *= exp(-dt * 0.8);
            p->vy *= exp(-dt * 0.5);
        }
    }

    /* Microwave popcorn: one pop per few forks, a DING and a fresh bag when it fills */
    corn_acc += dt * (y->forks_s > 0.5 ? fmin(10, 1.6 * log2(1 + y->forks_s / 15)) : 0);
    while (corn_acc >= 1) {
        corn_acc -= 1;
        for (int i = 0; i < MAX_CORN; i++)
            if (!corn[i].alive) {
                corn[i] = (kernel){ 1, 0, MW_X + 14 + frand() * 60, MW_Y + 44, (frand() - 0.5) * 50, -90 - 70 * frand(),
                                    frand() * 6 };
                break;
            }
    }
    int resting = 0;
    for (int i = 0; i < MAX_CORN; i++) {
        kernel *k = &corn[i];
        if (!k->alive)
            continue;
        if (!k->rest) {
            k->vy += 420 * dt;
            k->x += k->vx * dt;
            k->y += k->vy * dt;
            k->ang += dt * 8;
            if (k->x < MW_X + 11 || k->x > MW_X + 75)
                k->vx = -k->vx, k->x = fmax(MW_X + 11, fmin(MW_X + 75, k->x));
            if (k->y < MW_Y + 14)
                k->vy = fabs(k->vy) * 0.5, k->y = MW_Y + 14;
            double floor_y = MW_Y + 45 - (i % 5) * 1.2;
            if (k->y > floor_y) {
                k->y = floor_y;
                if (k->vy < 70)
                    k->rest = 1;
                k->vy *= -0.35;
                k->vx *= 0.5;
            }
        } else {
            resting++;
        }
    }
    if (resting >= MAX_CORN - 6) {
        memset(corn, 0, sizeof(corn));
        ding_t = t;
    }

    bread_a += ((gpu_idle && idle_for > 0.5 && piles[0].n == 0) - bread_a) * fmin(1, dt * 2.5);

    /* Moving parts */
    fan_ang += dt * sh->fan * 0.30;
    meter_phase += dt * sh->watts / 380.0;
    table_ang += dt * (0.4 + 3.5 * sh->disk) * (sh->disk > 0.02);

    /* The fly: buzzes about while the toasters are cold, leaves when work starts */
    fly_retarget -= dt;
    if (fly_retarget <= 0) {
        fly_retarget = 0.4 + frand() * 1.2;
        if (gpu_idle && idle_for > 1.5) {
            if (frand() < 0.25)
                fly_tx = plate_cx[0] + 4, fly_ty = PILE_BASE - 28;        /* visit the bread */
            else
                fly_tx = 110 + frand() * 260, fly_ty = 220 + frand() * 110;
        } else {
            fly_tx = 560, fly_ty = 120;
        }
    }
    double ax = (fly_tx - fly_x) * 6 - fly_vx * 2.6 + (frand() - 0.5) * 2200;
    double ay = (fly_ty - fly_y) * 6 - fly_vy * 2.6 + (frand() - 0.5) * 2200;
    fly_vx += ax * dt;
    fly_vy += ay * dt;
    fly_x += fly_vx * dt;
    fly_y += fly_vy * dt;
    if (fly_x > 540)
        fly_x = 540, fly_vx = 0;
}

/* ---------------------------------------------------------------- render */

static void draw_item(cairo_t *cr, int kind, int lvl, double x, double y, double ang, double sc, double squash)
{
    cairo_surface_t *img = kind == K_BAGEL ? bagel_img[lvl] : kind == K_WAFFLE ? waffle_img[lvl] : toast_img[lvl];
    double w = cairo_image_surface_get_width(img), h = cairo_image_surface_get_height(img);
    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, 1, squash);
    cairo_rotate(cr, ang);
    cairo_scale(cr, sc, sc);
    cairo_set_source_surface(cr, img, -w / 2, -h / 2);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);
    cairo_restore(cr);
}

/* Piles are cached: redrawn only when a slice lands or the pile topples */
static void update_pile(int g)
{
    pile_t *p = &piles[g];
    if (!p->dirty)
        return;
    p->dirty = 0;
    cairo_t *c = cairo_create(p->cache);
    cairo_set_operator(c, CAIRO_OPERATOR_CLEAR);
    cairo_paint(c);
    cairo_set_operator(c, CAIRO_OPERATOR_OVER);
    for (int i = 0; i < p->n; i++) {
        const pile_slice *s = &p->s[i];
        double x = 90 + s->dx, y = 140 - i * PILE_STEP;
        double sq = s->kind == K_TOAST ? 0.42 : 0.55;
        /* a thin dark edge under each slice gives the stack its layers */
        cairo_save(c);
        cairo_translate(c, x, y + 2.2);
        cairo_scale(c, 1, sq);
        cairo_rotate(c, s->ang);
        cairo_scale(c, s->scale * 1.02, s->scale * 1.02);
        cairo_surface_t *img = s->kind == K_BAGEL ? bagel_img[s->lvl] : s->kind == K_WAFFLE ? waffle_img[s->lvl] : toast_img[s->lvl];
        double w = cairo_image_surface_get_width(img), h = cairo_image_surface_get_height(img);
        cairo_set_source_rgba(c, 0.22, 0.12, 0.05, 0.55);
        cairo_mask_surface(c, img, -w / 2, -h / 2);
        cairo_restore(c);
        draw_item(c, s->kind, s->lvl, x, y, s->ang, s->scale, sq);
    }
    cairo_destroy(c);
}

/* Seven-segment digits for the kitchen timer */
static void seg_digit(cairo_t *cr, double x, double y, double w, double h, int d)
{
    static const unsigned char segs[10] = { 0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f };
    double t = 4.6, g = 0.9, hh = h / 2;
    /* a..g: top, top-right, bottom-right, bottom, bottom-left, top-left, middle */
    double sx[7][2] = { { x, y }, { x + w, y }, { x + w, y + hh }, { x, y + h }, { x, y + hh }, { x, y }, { x, y + hh } };
    int horiz[7] = { 1, 0, 0, 1, 0, 0, 1 };
    unsigned char m = d < 0 ? 0x7f : segs[d];
    for (int s = 0; s < 7; s++) {
        if (!(m >> s & 1))
            continue;
        double px = sx[s][0], py = sx[s][1];
        cairo_new_path(cr);
        if (horiz[s]) {
            double l = px + g, r = px + w - g;
            cairo_move_to(cr, l, py);
            cairo_line_to(cr, l + t / 2, py - t / 2);
            cairo_line_to(cr, r - t / 2, py - t / 2);
            cairo_line_to(cr, r, py);
            cairo_line_to(cr, r - t / 2, py + t / 2);
            cairo_line_to(cr, l + t / 2, py + t / 2);
        } else {
            double tp = py + g, bt = py + hh - g;
            cairo_move_to(cr, px, tp);
            cairo_line_to(cr, px + t / 2, tp + t / 2);
            cairo_line_to(cr, px + t / 2, bt - t / 2);
            cairo_line_to(cr, px, bt);
            cairo_line_to(cr, px - t / 2, bt - t / 2);
            cairo_line_to(cr, px - t / 2, tp + t / 2);
        }
        cairo_close_path(cr);
        cairo_fill(cr);
    }
}

static void update_lcd(const shown_t *sh, double t)
{
    static double val, shown_val = -1, next;
    char key[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        val = shown_rate(sh->tok);
    }
    /* hysteresis: only move the number when it has really changed */
    if (shown_val < 0 || fabs(val - shown_val) >= 0.9 || (val < 0.5 && shown_val != 0))
        shown_val = val < 0.5 ? 0 : round(val);
    int v = (int)shown_val, nd = gpu_source ? 3 : 4;
    if (v > (gpu_source ? 100 : 9999))
        v = gpu_source ? 100 : 9999;
    snprintf(key, sizeof(key), "%d|%d", v, gpu_source);
    if (!strcmp(key, lcd_key))
        return;
    strcpy(lcd_key, key);

    cairo_t *c = cairo_create(lcd_cache);
    cairo_set_operator(c, CAIRO_OPERATOR_CLEAR);
    cairo_paint(c);
    cairo_set_operator(c, CAIRO_OPERATOR_OVER);
    const char *unit = rate_unit();
    cairo_text_extents_t e;
    cairo_select_font_face(c, "DejaVu Sans Condensed", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, 22);
    cairo_text_extents(c, unit, &e);
    double dw = 20, pitch = 25, dh = 30;
    double total = nd * pitch - (pitch - dw) + 8 + e.x_advance;
    double x0 = (LCD_W - total) / 2, y0 = (LCD_H - dh) / 2;
    cairo_save(c);
    /* a slight italic, like a real LCD */
    cairo_matrix_t sk = { 1, 0, -0.08, 1, 0, 0 };
    cairo_translate(c, 0, LCD_H / 2.0);
    cairo_transform(c, &sk);
    cairo_translate(c, 0, -LCD_H / 2.0);
    char digits[16];
    snprintf(digits, sizeof(digits), "%*d", nd, v);
    for (int i = 0; i < nd; i++) {
        double x = x0 + i * pitch + 2;
        cairo_set_source_rgba(c, 0.10, 0.16, 0.10, 0.07);           /* ghost segments */
        seg_digit(c, x, y0, dw - 2, dh, -1);
        if (digits[i] != ' ') {
            cairo_set_source_rgba(c, 0.10, 0.14, 0.10, 0.92);
            seg_digit(c, x, y0, dw - 2, dh, digits[i] - '0');
        }
    }
    cairo_restore(c);
    cairo_set_source_rgba(c, 0.10, 0.14, 0.10, 0.9);
    cairo_move_to(c, x0 + nd * pitch - (pitch - dw) + 8 - e.x_bearing, y0 + dh + 1);
    cairo_show_text(c, unit);
    cairo_destroy(c);
}

/* Text on the cabinet door: total watts, CPU temperature, RAM */
static void update_text(const shown_t *sh, const sys_stats *y, double t)
{
    static double watts, temp, ram, next;
    char key[96], l1[48], l2[48];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        if (fabs(sh->watts - watts) >= 8)
            watts = sh->watts;
        if (fabs(sh->cpu_temp - temp) >= 0.8)
            temp = sh->cpu_temp;
        if (fabs(sh->ram * y->ram_total - ram) >= 0.4)
            ram = sh->ram * y->ram_total;
    }
    snprintf(l1, sizeof(l1), "%.2f kW · CPU %.0f°", watts / 1000.0, temp);
    snprintf(l2, sizeof(l2), "RAM %.0f GB", ram);
    snprintf(key, sizeof(key), "%s|%s", l1, l2);
    if (!strcmp(key, text_key))
        return;
    strcpy(text_key, key);

    cairo_t *c = cairo_create(text_cache);
    cairo_set_operator(c, CAIRO_OPERATOR_CLEAR);
    cairo_paint(c);
    cairo_set_operator(c, CAIRO_OPERATOR_OVER);
    const char *ls[2] = { l1, l2 };
    double ys[2] = { 18, 45 }, maxw[2] = { 290, 200 };     /* chord widths at y 430 and 457 */
    cairo_select_font_face(c, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    for (int i = 0; i < 2; i++) {
        cairo_text_extents_t e;
        double size = 22;
        cairo_set_font_size(c, size);
        cairo_text_extents(c, ls[i], &e);
        if (e.x_advance > maxw[i]) {
            size = fmax(18, size * maxw[i] / e.x_advance);
            cairo_set_font_size(c, size);
            cairo_text_extents(c, ls[i], &e);
        }
        cairo_move_to(c, 160 - e.width / 2 - e.x_bearing, ys[i] - e.height / 2 - e.y_bearing);
        cairo_text_path(c, ls[i]);
        cairo_set_line_join(c, CAIRO_LINE_JOIN_ROUND);
        cairo_set_line_width(c, 4.5);
        cairo_set_source_rgba(c, 1, 0.97, 0.88, 0.85);
        cairo_stroke_preserve(c);
        cairo_set_source_rgb(c, 0.36, 0.20, 0.10);
        cairo_fill(c);
    }
    cairo_destroy(c);
}

static void update_cookies(double frac)
{
    int want = (int)round(frac * 24);
    if (cookie_n >= 0 && abs(want - cookie_n) < 1)
        return;
    /* hysteresis: need a clear half-cookie change before redrawing */
    if (cookie_n >= 0 && fabs(frac * 24 - cookie_n) < 0.7)
        return;
    cookie_n = want;
    cairo_t *c = cairo_create(cookie_cache);
    cairo_set_operator(c, CAIRO_OPERATOR_CLEAR);
    cairo_paint(c);
    cairo_set_operator(c, CAIRO_OPERATOR_OVER);
    for (int i = 0; i < cookie_n; i++) {
        int row = i / 3, col = i % 3;
        double x = 14 + col * 13 + (row & 1) * 6 + ((i * 37) % 5 - 2), y = 72 - row * 7.2 + ((i * 53) % 3 - 1);
        cairo_save(c);
        cairo_translate(c, x, y);
        cairo_rotate(c, (i * 1.7));
        cairo_scale(c, 0.9, 0.9);
        cairo_set_source_surface(c, cookie_img, -8, -8);
        cairo_paint(c);
        cairo_restore(c);
    }
    cairo_destroy(c);
}

static void draw_flame(cairo_t *cr, double x, double y, double h, double w)
{
    cairo_new_path(cr);
    cairo_move_to(cr, x - w, y);
    cairo_curve_to(cr, x - w, y - h * 0.45, x - w * 0.2, y - h * 0.7, x, y - h);
    cairo_curve_to(cr, x + w * 0.2, y - h * 0.7, x + w, y - h * 0.45, x + w, y);
    cairo_close_path(cr);
    cairo_fill(cr);
}

/* 32 flames round the burner, one per CPU thread; back ones before the kettle, front ones after */
static void draw_flames(cairo_t *cr, const shown_t *sh, int n, int front, double t)
{
    const double cx = 240, cy = HOB_Y + 17, rx = 19, ry = 4.5;
    for (int i = 0; i < n; i++) {
        double a = (i + 0.5) / n * 2 * M_PI, sn = sin(a);
        if ((sn > 0) != front)
            continue;
        double l = sh->thread[i];
        double h = 2.5 + 20 * l + (l > 0.05 ? 2.0 * sin(t * 23 + i * 2.1) : 0.4 * sin(t * 9 + i));
        double x = cx + rx * cos(a), y = cy + ry * sn;
        cairo_set_source_rgba(cr, 0.25, 0.45, 1.0, 0.55 + 0.35 * l);
        draw_flame(cr, x, y, h, 2.6);
        cairo_set_source_rgba(cr, 0.75, 0.9, 1.0, 0.5 + 0.4 * l);
        draw_flame(cr, x, y, h * 0.5, 1.3);
    }
}

static void draw_kettle(cairo_t *cr, const shown_t *sh, double t)
{
    double w = sh->whistle, p = clamp01(sh->psi / 15);
    double wob = w > 0.02 ? 0.03 * w * sin(t * 50) : 0;
    double lift = (2.6 * p + 1.4 * w) * fabs(sin(t * 27)), tilt = 0.08 * p * sin(t * 19);
    const double lx = 25, ly = 17, lw = 46, lh = 21;      /* lid, in sprite coordinates */

    cairo_save(cr);
    cairo_translate(cr, KETTLE_X + 42, KETTLE_Y + 84);
    cairo_rotate(cr, wob);
    cairo_translate(cr, -42, -84);
    /* body, minus the lid */
    cairo_save(cr);
    cairo_rectangle(cr, -10, -10, 110, 110);
    cairo_rectangle(cr, lx, ly, lw, lh);
    cairo_set_fill_rule(cr, CAIRO_FILL_RULE_EVEN_ODD);
    cairo_clip(cr);
    blit(cr, kettle_img, 0, 0);
    cairo_restore(cr);
    /* the lid, rattling */
    cairo_save(cr);
    cairo_translate(cr, lx + lw / 2, ly + lh - lift);
    cairo_rotate(cr, tilt);
    cairo_translate(cr, -(lx + lw / 2), -(ly + lh));
    cairo_rectangle(cr, lx, ly - 4, lw, lh + 4);
    cairo_clip(cr);
    blit(cr, kettle_img, 0, 0);
    cairo_restore(cr);
    cairo_restore(cr);

    /* whistle: sound arcs off the spout */
    if (w > 0.02) {
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(cr, 2.2);
        for (int k = 0; k < 3; k++) {
            double ph = fmod(t * 3 + k / 3.0, 1.0);
            cairo_new_path(cr);
            cairo_arc(cr, KETTLE_X + 3, KETTLE_Y + 22, 6 + ph * 22, M_PI * 0.85, M_PI * 1.35);
            cairo_set_source_rgba(cr, 1, 1, 1, w * (1 - ph) * 0.95);
            cairo_stroke(cr);
        }
    }
}

static void draw_toaster(cairo_t *cr, int g, const shown_t *sh, double t)
{
    double bounce = -2.2 * exp(-(t - last_pop[g]) * 14) * (t >= last_pop[g]);
    double gl = sh->glow[g];
    blit(cr, toaster_img[g], toaster_x[g], toaster_y[g] + bounce);
    if (gl < 0.01)
        return;
    /* glowing elements seen down the slots */
    cairo_save(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int k = 0; k < 2; k++) {
        const double *r = slot_rect[g][k];
        double x = toaster_x[g] + r[0], y = toaster_y[g] + bounce + r[1];
        cairo_pattern_t *p = cairo_pattern_create_linear(0, y + r[3], 0, y);
        cairo_pattern_add_color_stop_rgba(p, 0, 1.0, 0.45, 0.08, 0.95 * gl);
        cairo_pattern_add_color_stop_rgba(p, 1, 1.0, 0.25, 0.02, 0.35 * gl);
        cairo_set_source(cr, p);
        cairo_rectangle(cr, x + 1, y + 1, r[2] - 2, r[3] - 1);
        cairo_fill(cr);
        cairo_pattern_destroy(p);
    }
    cairo_restore(cr);
    /* warm haze over the top of the toaster */
    double cx = toaster_x[g] + 68, cy = toaster_y[g] + 6;
    cairo_save(cr);
    cairo_translate(cr, cx, cy);
    cairo_scale(cr, 62, 26);
    cairo_pattern_t *p = cairo_pattern_create_radial(0, 0, 0, 0, 0, 1);
    double flick = 0.9 + 0.1 * sin(t * 13 + g * 2);
    cairo_pattern_add_color_stop_rgba(p, 0, 1.0, 0.55, 0.15, 0.42 * gl * flick);
    cairo_pattern_add_color_stop_rgba(p, 1, 1.0, 0.45, 0.10, 0);
    cairo_set_source(cr, p);
    cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_pattern_destroy(p);
    cairo_restore(cr);
}

static void draw_part(cairo_t *cr, const part *p)
{
    double k = p->age / p->life;
    if (p->kind == PT_NOTE_LAN || p->kind == PT_NOTE_TS) {
        cairo_surface_t *img = note_img[p->kind - PT_NOTE_LAN][p->glyph];
        double a = k < 0.15 ? k / 0.15 : 1 - (k - 0.15) / 0.85;
        double x = p->x + 7 * sin(p->age * 3 + p->seed);
        cairo_set_source_surface(cr, img, x - cairo_image_surface_get_width(img) / 2.0,
                                 p->y - cairo_image_surface_get_height(img) / 2.0);
        cairo_paint_with_alpha(cr, a);
        return;
    }
    double s = p->s0 + (p->s1 - p->s0) * sqrt(k), a = p->a0 * (1 - k) * fmin(1, k * 6);
    int r = (int)lround(s);
    r = r < 1 ? 1 : r > N_PUFF ? N_PUFF : r;
    cairo_set_source_surface(cr, puff_spr[p->kind == PT_SMOKE][r], (int)lround(p->x) - r, (int)lround(p->y) - r);
    cairo_paint_with_alpha(cr, a);
}

static void render(cairo_t *cr, const stats *s, const sys_stats *y, const shown_t *sh, double t)
{
    blit(cr, static_layer, 0, 0);

    /* Extractor fan: blades behind the grille, speed from the GPU fans */
    {
        double cx = FAN_X + 36, cy = FAN_Y + 34.5, spd = sh->fan / 100;
        cairo_save(cr);
        cairo_arc(cr, cx, cy, 26, 0, 2 * M_PI);
        cairo_clip(cr);
        cairo_set_source_rgb(cr, 0.07, 0.07, 0.08);
        cairo_paint(cr);
        for (int b = 0; b < 5; b++) {
            double a = fan_ang + b * 2 * M_PI / 5;
            cairo_save(cr);
            cairo_translate(cr, cx, cy);
            cairo_rotate(cr, a);
            cairo_scale(cr, 1, 0.42);
            cairo_arc(cr, 13, 0, 12, 0, 2 * M_PI);
            cairo_restore(cr);
            cairo_set_source_rgba(cr, 0.55, 0.56, 0.58, 1 - 0.55 * spd);
            cairo_fill(cr);
        }
        if (spd > 0.05) {                      /* motion blur */
            cairo_arc(cr, cx, cy, 25, 0, 2 * M_PI);
            cairo_set_source_rgba(cr, 0.45, 0.46, 0.5, 0.35 * spd);
            cairo_fill(cr);
        }
        cairo_arc(cr, cx, cy, 4, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.7, 0.7, 0.72);
        cairo_fill(cr);
        cairo_restore(cr);
        blit(cr, fan_img, FAN_X, FAN_Y);
    }

    /* Electricity meter: the disc's mark goes round with total watts */
    {
        double sx = METER_X + 13.5, sw = 36, sy = METER_Y + 28.5, shh = 7.5;
        double ph = fmod(meter_phase, 1.0);
        cairo_save(cr);
        cairo_rectangle(cr, sx, sy, sw, shh);
        cairo_clip(cr);
        for (int k = -1; k <= 1; k++) {
            double u = ph + k;                              /* 0..1 across the visible front */
            double x = sx + sw * (0.5 - 0.5 * cos(u * M_PI));
            double wdt = 1.5 + 3.5 * sin(u * M_PI);
            cairo_rectangle(cr, x - wdt / 2, sy, wdt, shh);
            cairo_set_source_rgba(cr, 0.75, 0.08, 0.05, 0.9);
            cairo_fill(cr);
        }
        cairo_restore(cr);
    }

    /* Microwave: the light and the turntable follow NVMe throughput; popcorn pops with forks */
    {
        double wx = MW_X + 7, wy = MW_Y + 11, ww = 72, wh = 38, d = sh->disk;
        double flash = clamp01(1 - (t - ding_t) / 0.5);
        cairo_save(cr);
        cairo_rectangle(cr, wx, wy, ww, wh);
        cairo_clip(cr);
        if (d > 0.01 || flash > 0) {
            cairo_pattern_t *p = cairo_pattern_create_radial(wx + ww / 2, wy + 4, 4, wx + ww / 2, wy + wh / 2, 48);
            cairo_pattern_add_color_stop_rgba(p, 0, 1.0, 0.86, 0.45, 0.85 * d + 0.6 * flash);
            cairo_pattern_add_color_stop_rgba(p, 1, 1.0, 0.70, 0.30, 0.35 * d + 0.3 * flash);
            cairo_set_source(cr, p);
            cairo_paint(cr);
            cairo_pattern_destroy(p);
        }
        /* glass turntable */
        double tx = wx + ww / 2, ty = wy + wh - 5;
        cairo_save(cr);
        cairo_translate(cr, tx, ty);
        cairo_scale(cr, 30, 5);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, 0.85, 0.85, 0.9, 0.18 + 0.25 * d);
        cairo_fill(cr);
        if (d > 0.02) {
            cairo_arc(cr, tx + 24 * cos(table_ang), ty + 4 * sin(table_ang), 2.2, 0, 2 * M_PI);
            cairo_set_source_rgba(cr, 1, 1, 1, 0.6 * d + 0.2);
            cairo_fill(cr);
        }
        for (int i = 0; i < MAX_CORN; i++) {
            const kernel *k = &corn[i];
            if (!k->alive)
                continue;
            for (int b = 0; b < 3; b++) {
                double a = k->ang + b * 2.1;
                cairo_arc(cr, k->x + 1.6 * cos(a), k->y + 1.6 * sin(a), 2.1, 0, 2 * M_PI);
            }
            cairo_set_source_rgb(cr, 1.0, 0.97, 0.86);
            cairo_fill(cr);
            cairo_arc(cr, k->x + 0.5, k->y + 0.8, 1.0, 0, 2 * M_PI);
            cairo_set_source_rgb(cr, 0.95, 0.72, 0.30);
            cairo_fill(cr);
        }
        cairo_restore(cr);
        if (d > 0.01 || flash > 0) {                       /* the little display lights up */
            cairo_rectangle(cr, MW_X + 92.5, MW_Y + 11.5, 14, 6);
            cairo_set_source_rgba(cr, 0.35, 1.0, 0.55, 0.25 + 0.6 * fmax(d, flash));
            cairo_fill(cr);
        }
    }

    /* Radio: the dial lights up and the needle swings with network traffic */
    {
        double dx = RADIO_X + 51, dy = RADIO_Y + 38, dw = 50, dh = 14, act = fmax(sh->net, sh->ts);
        cairo_rectangle(cr, dx, dy, dw, dh);
        cairo_set_source_rgba(cr, 1.0, 0.75, 0.3, 0.08 + 0.3 * act);
        cairo_save(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        cairo_fill(cr);
        cairo_restore(cr);
        double nx = dx + 3 + needle * (dw - 6);
        cairo_move_to(cr, nx, dy + 1);
        cairo_line_to(cr, nx, dy + dh - 1);
        cairo_set_line_width(cr, 1.6);
        cairo_set_source_rgb(cr, 0.80, 0.10, 0.06);
        cairo_stroke(cr);
        /* the speaker thumps */
        if (act > 0.05) {
            double r = 18 + 2.5 * act * fabs(sin(t * 9));
            cairo_arc(cr, RADIO_X + 26, RADIO_Y + 46, r, 0, 2 * M_PI);
            cairo_set_source_rgba(cr, 1.0, 0.9, 0.6, 0.18 * act);
            cairo_set_line_width(cr, 2);
            cairo_stroke(cr);
        }
    }

    /* Kitchen timer: the big number */
    update_lcd(sh, t);
    blit(cr, lcd_cache, LCD_X, LCD_Y);

    /* Smoke alarm: LED blips every few seconds, flashes and beeps when the GPUs cook */
    {
        double lx = 240, ly = ALARM_Y + 7.5;
        if (alarm_on) {
            double at = t - alarm_t, on = fmod(at * 3, 1.0) < 0.5;
            cairo_pattern_t *p = cairo_pattern_create_radial(lx, ly, 0, lx, ly, 34);
            cairo_pattern_add_color_stop_rgba(p, 0, 1, 0.15, 0.1, on ? 0.75 : 0.25);
            cairo_pattern_add_color_stop_rgba(p, 1, 1, 0.1, 0.1, 0);
            cairo_set_source(cr, p);
            cairo_arc(cr, lx, ly, 34, 0, 2 * M_PI);
            cairo_fill(cr);
            cairo_pattern_destroy(p);
            cairo_arc(cr, lx, ly, 3, 0, 2 * M_PI);
            cairo_set_source_rgb(cr, 1, on ? 0.5 : 0.1, on ? 0.4 : 0.1);
            cairo_fill(cr);
            /* sound rings */
            cairo_set_line_width(cr, 2.5);
            for (int k = 0; k < 2; k++) {
                double ph = fmod(at * 1.5 + k * 0.5, 1.0);
                cairo_new_path(cr);
                cairo_arc(cr, 240, ALARM_Y + 29, 32 + ph * 30, 0, 2 * M_PI);
                cairo_set_source_rgba(cr, 1, 0.2, 0.15, 0.55 * (1 - ph));
                cairo_stroke(cr);
            }
            int side = (int)(at * 1.5) & 1;
            double pop = fmod(at * 1.5, 1.0), sc = 0.8 + 0.2 * fmin(1, pop * 6);
            double bw = cairo_image_surface_get_width(beep_img), bh = cairo_image_surface_get_height(beep_img);
            cairo_save(cr);
            cairo_translate(cr, side ? 318 : 162, ALARM_Y + 40 - 4 * sin(pop * M_PI));
            cairo_rotate(cr, side ? 0.12 : -0.12);
            cairo_scale(cr, sc, sc);
            cairo_set_source_surface(cr, beep_img, -bw / 2, -bh / 2);
            cairo_paint(cr);
            cairo_restore(cr);
        } else if (fmod(t, 4.0) < 0.12) {
            cairo_arc(cr, lx, ly, 2.2, 0, 2 * M_PI);
            cairo_set_source_rgb(cr, 1, 0.25, 0.2);
            cairo_fill(cr);
        }
    }

    /* Kettle on the gas ring */
    int nt = y->n_threads > 0 ? y->n_threads : 32;
    if (nt > SYS_MAX_THREADS)
        nt = SYS_MAX_THREADS;
    {
        /* blue glow under the kettle */
        cairo_save(cr);
        cairo_translate(cr, 240, HOB_Y + 12);
        cairo_scale(cr, 34, 12);
        cairo_pattern_t *p = cairo_pattern_create_radial(0, 0, 0, 0, 0, 1);
        cairo_pattern_add_color_stop_rgba(p, 0, 0.35, 0.55, 1.0, 0.15 + 0.45 * sh->cpu);
        cairo_pattern_add_color_stop_rgba(p, 1, 0.35, 0.55, 1.0, 0);
        cairo_set_source(cr, p);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(p);
        cairo_restore(cr);
    }
    draw_flames(cr, sh, nt, 0, t);
    draw_kettle(cr, sh, t);
    draw_flames(cr, sh, nt, 1, t);

    /* Toasters, and the toast rising out of their slots */
    for (int g = 0; g < 2; g++) {
        draw_toaster(cr, g, sh, t);
        for (int i = 0; i < MAX_FLYERS; i++) {
            const flyer *f = &flyers[i];
            if (!f->alive || f->phase != P_RISE || f->gpu != g)
                continue;
            cairo_save(cr);
            cairo_rectangle(cr, 0, 0, SIZE, f->clip);
            cairo_clip(cr);
            draw_item(cr, f->kind, f->lvl, f->x, f->y, 0, f->scale, 1);
            cairo_restore(cr);
        }
    }

    /* Cookie jar: RAM in use */
    update_cookies(clamp01(sh->ram));
    blit(cr, cookie_cache, JAR_X, JAR_Y);
    blit(cr, jar_img, JAR_X, JAR_Y);

    /* Plates, piles and the waiting slice */
    for (int g = 0; g < 2; g++) {
        blit(cr, plate_img, plate_cx[g] - 58, PLATE_Y);
        update_pile(g);
        blit(cr, piles[g].cache, plate_cx[g] - 90, PILE_BASE - 140);
    }
    if (bread_a > 0.01) {
        /* it drops onto the plate as it fades in */
        cairo_save(cr);
        cairo_rectangle(cr, plate_cx[0] - 30, PILE_BASE - 60, 60, 64);
        cairo_clip(cr);
        cairo_push_group(cr);
        draw_item(cr, K_TOAST, 0, plate_cx[0] + 4, PILE_BASE - 16 - 12 * (1 - bread_a), -0.12, 0.95, 1);
        cairo_pop_group_to_source(cr);
        cairo_paint_with_alpha(cr, bread_a);
        cairo_restore(cr);
    }

    /* Toast in flight, and toppled piles on their way to the floor */
    for (int i = 0; i < MAX_FLYERS; i++) {
        const flyer *f = &flyers[i];
        if (f->alive && f->phase != P_RISE)
            draw_item(cr, f->kind, f->lvl, f->x, f->y, f->ang, f->scale, 1);
    }

    for (int i = 0; i < MAX_PARTS; i++)
        if (parts[i].alive)
            draw_part(cr, &parts[i]);

    /* The fly */
    if (fly_x > -20 && fly_x < SIZE + 20) {
        double wing = fabs(sin(t * 60));
        cairo_save(cr);
        cairo_translate(cr, fly_x, fly_y);
        cairo_rotate(cr, atan2(fly_vy, fly_vx) * 0.25);
        cairo_scale(cr, 1.3, 1.3);
        cairo_set_source_rgba(cr, 0.85, 0.92, 1.0, 0.55);
        for (int k = -1; k <= 1; k += 2) {
            cairo_save(cr);
            cairo_translate(cr, -1, -3);
            cairo_rotate(cr, k * (0.5 + 0.6 * wing));
            cairo_scale(cr, 2.4, 5);
            cairo_arc(cr, 0, -0.8, 1, 0, 2 * M_PI);
            cairo_restore(cr);
            cairo_fill(cr);
        }
        cairo_save(cr);
        cairo_scale(cr, 4.6, 3.4);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgb(cr, 0.10, 0.10, 0.12);
        cairo_fill(cr);
        cairo_arc(cr, 3.8, -0.5, 2.2, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.45, 0.08, 0.06);
        cairo_fill(cr);
        cairo_restore(cr);
    }

    update_text(sh, y, t);
    blit(cr, text_cache, 80, 412);
    (void)s;
}

/* ---------------------------------------------------------------- smoothing */

static double disk_act(const sys_stats *y)
{
    double b = 0;
    for (int i = 0; i < N_NVME; i++)
        b += y->nvme_rd[i] + y->nvme_wr[i];
    return clamp01(log10(1 + b / 1e6) / log10(1 + 3000));          /* 0 .. 3 GB/s, log */
}

static double net_act(double bytes)
{
    return clamp01(log10(1 + bytes / 1e3) / log10(1 + 1.2e5));        /* 0 .. 120 MB/s, log */
}

static void ease(const stats *s, const sys_stats *y, shown_t *sh, double dt)
{
    double k = fmin(1, dt * 4), slow = fmin(1, dt * 1.2);
    sh->tok += (s->tok_s - sh->tok) * k;
    for (int g = 0; g < 2; g++) {
        sh->rate[g] += (s->tok_port[g] - sh->rate[g]) * k;
        sh->glow[g] += (clamp01((s->power[g] - GPU_IDLE_W) / (GPU_MAX_W - 2 * GPU_IDLE_W)) - sh->glow[g]) * k;
        sh->done[g] += (clamp01((s->temp[g] - 32) / 48.0) - sh->done[g]) * slow;
    }
    sh->cpu += (y->cpu_load - sh->cpu) * k;
    for (int i = 0; i < SYS_MAX_THREADS; i++)
        sh->thread[i] += (y->thread_load[i] - sh->thread[i]) * fmin(1, dt * 6);
    sh->cpu_temp += (y->cpu_temp - sh->cpu_temp) * k;
    sh->whistle += (clamp01((y->cpu_temp - 78) / 8) - sh->whistle) * k;
    sh->psi += (y->psi_cpu - sh->psi) * k;
    sh->ram += ((y->ram_total > 0 ? y->ram_used / y->ram_total : 0) - sh->ram) * slow;
    sh->disk += (disk_act(y) - sh->disk) * k;
    sh->net += (net_act(y->net_rx + y->net_tx) - sh->net) * k;
    sh->ts += (net_act(y->ts_rx + y->ts_tx) - sh->ts) * k;
    sh->fan += ((s->fan[0] + s->fan[1]) / 2 - sh->fan) * k;
    sh->watts += (s->power[0] + s->power[1] + y->pkg_watts - sh->watts) * k;
    needle += (fmax(sh->net, sh->ts) - needle) * fmin(1, dt * 2);
}

static void snap(const stats *s, const sys_stats *y, shown_t *sh)
{
    ease(s, y, sh, 10);
    needle = fmax(sh->net, sh->ts);
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats s = { 0 };
    sys_stats y = { 0 };
    shown_t sh = { 0 };
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
    load_assets();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* scenes from the showcase script: flat out and burnt, busy and golden, idle */
        struct { double at; const char *png; } scenes[] = {
            { 22.5, "toaster_preview.png" }, { 14.5, "toaster_mid.png" }, { 3.5, "toaster_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(7 + k);
            int n = 10 * FPS_BUSY;
            size_t len = 0;
            double b0 = 0, start = scenes[k].at - n / (double)FPS_BUSY;
            for (int i = 0; i < n; i++) {
                double t = start + i / (double)FPS_BUSY;
                if (i == n - 60)
                    b0 = now_s();
                showcase_poll(&s, &y, t);
                if (i == 0)
                    snap(&s, &y, &sh);
                ease(&s, &y, &sh, 1.0 / FPS_BUSY);
                simulate(&s, &y, &sh, 1.0 / FPS_BUSY, t);
                render(cr, &s, &y, &sh, t);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / 60, len);
            cairo_surface_write_to_png(surf, scenes[k].png);
        }
        return 0;
    }

    if (!demo) {
        gpus_init();
        sys_init();
    }

    while ((fd = lcd_open()) < 0 && !stop)
        sleep(2);
    if (fd >= 0)
        lcd_brightness(fd, 100);

    t0 = last = now_s();
    int first = 1;
    while (!stop) {
        double t = now_s(), dt = t - last;
        last = t;
        if (dt > 0.25)
            dt = 0.25;

        if (showcase) {
            showcase_poll(&s, &y, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 0.5;
            if (demo) {
                /* fresh values from the script each time; nothing is scaled in place */
                demo_poll(&s, &y, t - t0);
            } else {
                gpus_poll(&s);
                sys_poll(&y, t);
                if (gpu_source)
                    gpu_rate_poll(&s);
                else
                    vllm_poll(&s, t);
            }
        }
        if (first) {
            snap(&s, &y, &sh);
            first = 0;
        }

        ease(&s, &y, &sh, dt);
        simulate(&s, &y, &sh, dt, t - t0);
        render(cr, &s, &y, &sh, t - t0);
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

        int idle = s.tok_s < 1 && s.running == 0 && !showcase;
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
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
