/*
 * shrine: a tiny cult worships your graphics cards, on the iCUE LINK AIO pump LCD.
 *
 * Hooded acolytes kneel in a ritual circle around the monolith (a graphics card
 * standing on its end on a stone altar) and bow in time with the LLM's tokens/sec
 * (or GPU activity in GPU mode): slow, reverent bows when it's quiet, arms-in-the-air
 * frenzy at the top. Every few tokens an acolyte sends up a glowing rune that flies
 * into the monolith. The whole machine takes part: the two braziers burn with each
 * GPU's power, 32 candles around the circle are the 32 CPU threads, the offering
 * bowl fills with RAM, the gong is struck by NVMe traffic, carrier pigeons fly
 * network traffic in and out, the prayer bell swings with interrupts, runes on the
 * floor flare as processes are forked, incense rises with the kernel's entropy,
 * zombie processes shamble around the back as undead acolytes, and more acolytes
 * shuffle in as the machine gets busier. At full heat (GPU or CPU) the monolith
 * glows red and an acolyte fans it with a palm leaf. Idle, one acolyte sweeps and
 * the rest nod off. The monolith's fans spin with the GPU fans.
 * The acolyte, monolith, props and temple floor were generated with an image model
 * (assets/shrine/); flames, glows, runes, smoke and text are drawn with cairo.
 * Run with --demo to simulate data, --showcase for a scripted 40 s service,
 * --bench to write preview PNGs.
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
    double fan[N_GPUS];         /* 0..1 */
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
            s->fan[i] = fan / 100.0;
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

/* One GPU's activity 0..1: half utilisation, half power between idle and limit */
static double gpu_activity(const stats *s, int i)
{
    double pw = clamp01((s->power[i] - GPU_IDLE_W) / (GPU_MAX_W - GPU_IDLE_W));
    double a = 0.5 * clamp01(s->load[i]) + 0.5 * pw;
    return a < 0.03 ? 0 : a;                /* no idle jitter */
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
        double a = gpu_activity(s, i);
        s->tok_port[i] = a * GPU_FULL_RATE;
        s->running_port[i] = a > 0.05 ? (int)ceil(a * 4) : 0;
        s->tok_s += a * GPU_FULL_RATE;
        s->running += a > 0.05;
    }
}

/* ---- system sensors ---- */

/*
 * The rest of the machine, straight from the kernel: per-thread CPU load, interrupts,
 * forks (/proc/stat), CPU temperature (hwmon "k10temp", found by name), RAM
 * (/proc/meminfo), NVMe throughput (/proc/diskstats), network (/proc/net/dev),
 * task count (/proc/loadavg), entropy pool, uptime, and zombie processes (state Z in
 * /proc/<pid>/stat, counted every 5 s). sys_poll() is cheap and meant for ~2 Hz; the
 * counters are differenced into rates. Missing sensors just stay at 0.
 */
#define SYS_MAX_THREADS 64
#define N_NVME          3

typedef struct {
    int    n_threads;
    double thread_load[SYS_MAX_THREADS];    /* 0..1 per hardware thread */
    double cpu_load;                        /* 0..1 whole package */
    double cpu_temp;                        /* k10temp Tctl, C */
    double ram_total, ram_used;             /* GB (used = total - available) */
    double disk_rd, disk_wr;                /* bytes/s, nvme0n1..nvme2n1 summed */
    double net_rx, net_tx;                  /* bytes/s, enp12s0 + tailscale0 */
    double intr_s, forks_s;                 /* interrupts/s, processes forked/s */
    int    tasks;                           /* all tasks (threads) in the system */
    int    zombies;                         /* processes in state Z */
    double entropy;                         /* entropy_avail, bits (0..256) */
    double uptime;                          /* s */
} sys_stats;

static const char *NET_IFS[] = { "enp12s0", "tailscale0" };
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

static int count_zombies(void)
{
    DIR *d = opendir("/proc");
    struct dirent *e;
    char path[300], buf[512];
    int n = 0;

    while (d && (e = readdir(d))) {
        if (e->d_name[0] < '1' || e->d_name[0] > '9')
            continue;
        snprintf(path, sizeof(path), "/proc/%s/stat", e->d_name);
        if (read_text(path, buf, sizeof(buf)) <= 0)
            continue;
        const char *p = strrchr(buf, ')');
        if (p && p[1] == ' ' && p[2] == 'Z')
            n++;
    }
    if (d)
        closedir(d);
    return n;
}

static void sys_poll(sys_stats *s, double t)
{
    static unsigned long long last_busy[SYS_MAX_THREADS], last_total[SYS_MAX_THREADS];
    static unsigned long long last_rd, last_wr, last_rx, last_tx, last_intr, last_forks;
    static double last_t, next_zombie;
    static int have;
    double dt = t - last_t;
    char line[512], buf[128];
    FILE *f;

    /* CPU per thread, interrupts and forks since the last poll */
    if ((f = fopen("/proc/stat", "r"))) {
        unsigned long long all_busy = 0, all_total = 0;
        int n = 0;
        while (fgets(line, sizeof(line), f)) {
            if (!strncmp(line, "cpu", 3) && line[3] != ' ') {
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
                if (id + 1 > n)
                    n = id + 1;
            } else if (!strncmp(line, "intr ", 5)) {
                unsigned long long v = strtoull(line + 5, NULL, 10);
                if (have && dt > 0 && v >= last_intr)
                    s->intr_s = (v - last_intr) / dt;
                last_intr = v;
            } else if (!strncmp(line, "processes ", 10)) {
                unsigned long long v = strtoull(line + 10, NULL, 10);
                if (have && dt > 0 && v >= last_forks)
                    s->forks_s = (v - last_forks) / dt;
                last_forks = v;
            }
        }
        fclose(f);
        s->n_threads = n;
        if (all_total)
            s->cpu_load = (double)all_busy / all_total;
    }

    if (*k10_temp_path && read_text(k10_temp_path, buf, sizeof(buf)) > 0)
        s->cpu_temp = strtod(buf, NULL) / 1000.0;

    if ((f = fopen("/proc/meminfo", "r"))) {
        double total = 0, avail = 0, v;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %lf", &v) == 1)
                total = v;
            else if (sscanf(line, "MemAvailable: %lf", &v) == 1) {
                avail = v;
                break;                                  /* comes after MemTotal */
            }
        }
        fclose(f);
        s->ram_total = total / 1048576.0;
        s->ram_used = (total - avail) / 1048576.0;
    }

    /* NVMe: sectors read (field 6) and written (field 10), 512 bytes each */
    if ((f = fopen("/proc/diskstats", "r"))) {
        unsigned long long rd_all = 0, wr_all = 0;
        while (fgets(line, sizeof(line), f)) {
            char name[32];
            unsigned long long rd, wr;
            if (sscanf(line, "%*u %*u %31s %*u %*u %llu %*u %*u %*u %llu", name, &rd, &wr) != 3)
                continue;
            if (strncmp(name, "nvme", 4) || strlen(name) != 7 || strcmp(name + 5, "n1") ||
                name[4] < '0' || name[4] >= '0' + N_NVME)
                continue;
            rd_all += rd;
            wr_all += wr;
        }
        fclose(f);
        if (have && dt > 0) {
            s->disk_rd = rd_all >= last_rd ? (rd_all - last_rd) * 512.0 / dt : 0;
            s->disk_wr = wr_all >= last_wr ? (wr_all - last_wr) * 512.0 / dt : 0;
        }
        last_rd = rd_all;
        last_wr = wr_all;
    }

    if ((f = fopen("/proc/net/dev", "r"))) {
        unsigned long long rx_all = 0, tx_all = 0;
        while (fgets(line, sizeof(line), f)) {
            char *colon = strchr(line, ':'), *name = line;
            unsigned long long rx, tx;
            if (!colon)
                continue;
            *colon = 0;
            while (*name == ' ')
                name++;
            if ((strcmp(name, NET_IFS[0]) && strcmp(name, NET_IFS[1])) ||
                sscanf(colon + 1, "%llu %*u %*u %*u %*u %*u %*u %*u %llu", &rx, &tx) != 2)
                continue;
            rx_all += rx;
            tx_all += tx;
        }
        fclose(f);
        if (have && dt > 0) {
            s->net_rx = rx_all >= last_rx ? (rx_all - last_rx) / dt : 0;
            s->net_tx = tx_all >= last_tx ? (tx_all - last_tx) / dt : 0;
        }
        last_rx = rx_all;
        last_tx = tx_all;
    }

    if (read_text("/proc/loadavg", buf, sizeof(buf)) > 0) {
        const char *slash = strchr(buf, '/');
        if (slash)
            s->tasks = atoi(slash + 1);
    }
    if (read_text("/proc/sys/kernel/random/entropy_avail", buf, sizeof(buf)) > 0)
        s->entropy = strtod(buf, NULL);
    if (read_text("/proc/uptime", buf, sizeof(buf)) > 0)
        s->uptime = strtod(buf, NULL);
    if (t >= next_zombie) {                                 /* ~700 small reads: only every 5 s */
        next_zombie = t + 5;
        s->zombies = count_zombies();
    }
    last_t = t;
    have = 1;
}

/* ---- end system sensors ---- */

/* ---------------------------------------------------------------- simulated data */

static double frand(void) { return rand() / (double)RAND_MAX; }

/*
 * Fill a sys_stats from a handful of 0..1 knobs (used by --demo and --showcase). The
 * threads get their own slow waves so the candle ring ripples instead of moving as one.
 */
static void sys_sim(sys_stats *s, double t, double cpu, double disk, double net, double ram, double hot, int zombies)
{
    double sum = 0;
    s->n_threads = 32;
    for (int i = 0; i < 32; i++) {
        double w = 0.5 + 0.5 * sin(t * (0.7 + 0.13 * (i % 7)) + i * 1.7);
        double l = cpu * (0.55 + 0.7 * w) + 0.03 * frand();
        if (i % 16 == 3 && cpu > 0.05)
            l += 0.35;                                      /* a couple of always-busy threads */
        s->thread_load[i] = clamp01(l);
        sum += s->thread_load[i];
    }
    s->cpu_load = sum / 32;
    s->cpu_temp = 42 + 50 * clamp01(0.7 * cpu + 0.5 * hot);
    s->ram_total = 91.9;
    s->ram_used = 91.9 * clamp01(ram);
    double d = disk > 0.01 ? 1e5 * pow(3e4, disk) : 0;     /* 100 KB/s .. 3 GB/s */
    s->disk_rd = d * (0.6 + 0.3 * sin(t * 0.9));
    s->disk_wr = d * (0.4 + 0.3 * sin(t * 1.3 + 1));
    double n = net > 0.01 ? 2e3 * pow(5e4, net) : 500;     /* 2 KB/s .. 100 MB/s */
    s->net_rx = n * (0.7 + 0.3 * sin(t * 0.8));
    s->net_tx = n * 0.35 * (0.7 + 0.3 * sin(t * 1.1 + 2));
    s->intr_s = 2.5e4 * pow(40, clamp01(0.2 + 0.8 * cpu)) * (0.9 + 0.2 * frand());
    s->forks_s = 1 + 60 * cpu * cpu + 4 * frand();
    s->tasks = 3300 + (int)(700 * cpu);
    s->zombies = zombies;
    s->entropy = 256;
    s->uptime = 3 * 86400 + 5 * 3600 + t;
}

static void demo_poll(stats *g, sys_stats *s, double t)
{
    double busy = 0.5 + 0.5 * sin(t * 0.21);
    double b2 = 0.5 + 0.5 * sin(t * 0.13 + 2);
    g->load[0] = clamp01(busy + (frand() - 0.5) * 0.1);
    g->load[1] = clamp01(busy * b2 + (frand() - 0.5) * 0.1);
    for (int i = 0; i < 2; i++) {
        g->power[i] = 30 + 520 * g->load[i];
        g->temp[i] = (int)(38 + 46 * g->load[i]);
        g->fan[i] = 0.3 + 0.6 * g->load[i];
        g->tok_port[i] = g->load[i] > 0.2 ? 900 * g->load[i] : 0;
        g->running_port[i] = g->tok_port[i] > 0 ? 2 : 0;
    }
    g->tok_s = g->tok_port[0] + g->tok_port[1];
    g->running = g->running_port[0] + g->running_port[1];
    sys_sim(s, t, 0.1 + 0.8 * (0.5 + 0.5 * sin(t * 0.17 + 1)), 0.5 + 0.5 * sin(t * 0.31),
            0.5 + 0.5 * sin(t * 0.23 + 3), 0.3 + 0.5 * (0.5 + 0.5 * sin(t * 0.05)), busy, (int)(1.5 + 1.5 * sin(t * 0.07)));
}

/*
 * --showcase: a scripted 40 s service. Night (one sweeper, the rest asleep), the CPU
 * wakes the candles and the gong, GPU 0 lights its brazier, GPU 1 joins, both flat
 * out into a frenzy, it overheats (red monolith, palm-leaf fanning), then it winds down.
 */
static void showcase_poll(stats *g, sys_stats *s, double t)
{
    double u = fmod(t, 40.0), a = 0, b = 0, cpu = 0.03, disk = 0, net = 0, hot = 0;
    int z = 0;
    if (u < 5) {
        cpu = 0.03;
    } else if (u < 10) {
        double k = clamp01((u - 5) / 2);
        cpu = 0.03 + 0.4 * k, disk = 0.75 * k, net = 0.3 * k, a = 120 * k;
    } else if (u < 16) {
        double k = clamp01((u - 10) / 2);
        a = 120 + 420 * k, b = 60 * k, cpu = 0.45, disk = 0.35, net = 0.3 + 0.4 * k, z = 1;
    } else if (u < 19) {
        double k = clamp01((u - 16) / 1.5);
        a = 540 + 320 * k, b = 60 + 780 * k, cpu = 0.5 + 0.35 * k, disk = 0.5, net = 0.8, z = 2;
    } else if (u < 30) {
        a = 860 + 30 * sin(u * 2), b = 840 + 30 * sin(u * 1.7), cpu = 0.9, disk = 0.9, net = 0.9, z = 2;
        hot = clamp01((u - 19) / 2);
    } else if (u < 34) {
        double k = 1 - clamp01((u - 30) / 3);
        a = 500 * k, b = 300 * k, cpu = 0.03 + 0.4 * k, disk = 0.3 * k, net = 0.3 * k, z = 1;
    }
    g->tok_port[0] = a > 0 ? a + (frand() - 0.5) * 20 : 0;
    g->tok_port[1] = b > 0 ? b + (frand() - 0.5) * 20 : 0;
    g->tok_s = g->tok_port[0] + g->tok_port[1];
    g->running_port[0] = a > 0 ? 1 + (int)(a / 200) : 0;
    g->running_port[1] = b > 0 ? 1 + (int)(b / 200) : 0;
    g->running = g->running_port[0] + g->running_port[1];
    for (int i = 0; i < 2; i++) {
        double l = clamp01(g->tok_port[i] / 880);
        g->load[i] = l;
        g->power[i] = 32 + 530 * l;
        g->temp[i] = (int)(36 + 30 * l + 20 * hot);
        g->fan[i] = 0.3 + 0.7 * l;
    }
    sys_sim(s, t, cpu, disk, net, 0.25 + 0.55 * clamp01((a + b) / 1700), hot, z);
}

/* ---------------------------------------------------------------- scene */

/*
 * Screen layout (480 x 480, round). The temple floor painting has the ritual circle
 * centred at (240, 290). The monolith stands in the middle, braziers either side,
 * the gong back-left and the prayer bell back-right, the offering bowl and two
 * censers in front of the altar, 32 candles on a ring just inside the carved rim and
 * the stone tablet at the bottom. Acolytes kneel on an inner ring facing the monolith.
 */
#define FPS_BUSY        20
#define FPS_IDLE        15
#define MONO_X          240.0
#define MONO_BASE       312.0       /* bottom of the altar */
#define MONO_K          0.5         /* asset -> screen */
#define BRZ_Y           238.0       /* braziers' feet */
static const double BRZ_X[2] = { 86, 394 };
#define GONG_X          168.0
#define GONG_Y          206.0
#define BELL_X          300.0       /* top left of the bell and bracket */
#define BELL_Y          118.0
#define BOWL_Y          346.0
#define CENSER_Y        345.0
static const double CENSER_X[2] = { 191, 289 };
#define RING_CX         240.0       /* candle ring */
#define RING_CY         292.0
#define RING_RX         214.0
#define RING_RY         98.0
#define N_CANDLES       32
#define TAB_W           212
#define TAB_H           94
#define TAB_X           (240 - TAB_W / 2)
#define TAB_Y           358
#define CAP_W           300
#define CAP_H           36
#define CAP_Y           24

/* acolytes */
#define MAX_ACO         8
#define ACO_MIN         4
#define N_POSE          12
#define N_ASC           10          /* pre-scaled sizes */
#define ASC_MIN         0.34
#define ASC_MAX         0.47
#define KNEEL_HALF      47.0        /* asset px from a kneeler's back to its middle */
#define WALK_SPEED      38.0
#define SWEEP_Y         357.0
#define SWEEP_X0        128.0
#define SWEEP_X1        352.0
#define FAN_X           322.0
#define FAN_Y           308.0
#define MAX_ZOMB        3
#define ZOMB_Y          210.0

/* sprite sheet: size and the anchor x (asset px): the back of the robe for kneeling
   poses, the middle of the feet for standing ones */
static const struct { int kneel; double ax; } POSE[N_POSE] = {
    { 1, 0.4 }, { 1, 0.0 }, { 1, 0.4 }, { 1, 10.5 },       /* upright, half bow, full bow, arms up */
    { 0, 50.0 }, { 0, 46.6 }, { 0, 72.9 }, { 1, 0.0 },     /* walk a, walk b, asleep curled, asleep sitting */
    { 0, 76.0 }, { 0, 58.4 }, { 1, 0.0 }, { 1, 0.0 },      /* sweep a, sweep b, fan up, fan down */
};
enum { P_UP, P_HALF, P_BOW, P_RAISE, P_WALK0, P_WALK1, P_CURL, P_SIT, P_SWEEP0, P_SWEEP1, P_FAN0, P_FAN1 };
enum { A_OFF, A_ENTER, A_KNEEL, A_SLEEP, A_TO_SWEEP, A_SWEEP, A_RETURN, A_LEAVE };

typedef struct {
    int    state, face, frame, prev, slot;
    double x, y, fade, fdur, walk, off, dir, sway;
} acolyte;

typedef struct { double x0, y0, cx, cy, x1, y1, t, dur; int g, alive; } glyph;
typedef struct { double x, y, vx, age, life, size; int alive; } puff;
typedef struct { double x, y, vx, vy, age, life; int alive; } spark;
typedef struct { double t, dur, y0, y1; int out, alive; } pigeon;
typedef struct { double age; int alive; } ring;

#define MAX_GLYPH   64
#define MAX_PUFF    72
#define MAX_SPARK   64
#define MAX_PIGEON  8
#define MAX_RING    8
#define N_GLYPH_SPR 8
#define N_RUNES     26
#define N_FLAME     18
#define N_PUFF_SPR  10

static acolyte aco[MAX_ACO], fanner, zomb[MAX_ZOMB];
static double  slot_x[MAX_ACO], slot_y[MAX_ACO];
static glyph   glyphs[MAX_GLYPH];
static puff    puffs[MAX_PUFF];
static spark   sparks[MAX_SPARK];
static pigeon  pigeons[MAX_PIGEON];
static ring    gong_rings[MAX_RING], bell_rings[MAX_RING];
static double  cand_x[N_CANDLES], cand_y[N_CANDLES], cand_l[N_CANDLES];
static double  rune_x[N_RUNES], rune_y[N_RUNES], rune_lit[N_RUNES];
static int     rune_g[N_RUNES];

static cairo_surface_t *bg_img, *mono_img, *mono_src, *brazier_img, *gong_img, *bell_img, *censer_img, *bowl_img;
static cairo_surface_t *candle_img[3], *pigeon_src[3], *tablet_img, *hud_cache, *cap_cache, *heap_img;
static cairo_surface_t *aco_spr[N_POSE][2][N_ASC], *zomb_spr[2][2];
static cairo_surface_t *glyph_spr[N_GLYPH_SPR], *rune_spr[N_GLYPH_SPR], *flame_spr[N_FLAME], *big_flame;
static cairo_surface_t *puff_spr[N_PUFF_SPR], *zzz_spr, *spark_spr, *shadow_spr;
static cairo_surface_t *halo_cool, *halo_hot, *fire_glow, *candle_glow, *heap_glow;
static double  flame_h[N_FLAME];

/* the living state of the service */
typedef struct {
    double act, mload, heat, pw[2], fanfrac, ram, disk, net_rx, net_tx, intr, forks, entropy;
    double chant_ph, pulse, fan_ang, gong_next, gong_flash, gong_shake, bell_ph, bell_amp;
    double spawn_glyph, spawn_fork, spawn_puff[2], spawn_spark[2], spawn_pig[2];
    double idle_t, count_t, hot_t, frenzy_x;
    int    idle, count, hot, frenzy, mood, zomb_n;
} world;
static world W;

static const char *RUNES[N_GLYPH_SPR] = { "\xe1\x9a\xa0", "\xe1\x9a\xa2", "\xe1\x9a\xa6", "\xe1\x9a\xb1",
                                          "\xe1\x9b\x9f", "\xe1\x9b\x89", "\xe1\x9b\x9e", "\xe1\x9a\xb9" };

static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/shrine/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/shrine/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "shrine: can't load %s (looked next to the binary and in ./assets/shrine)\n", name);
        exit(1);
    }
    return s;
}

static int sw(cairo_surface_t *s) { return cairo_image_surface_get_width(s); }
static int sh_(cairo_surface_t *s) { return cairo_image_surface_get_height(s); }

/* A copy of src scaled by k (and mirrored), resampled once with the best filter */
static cairo_surface_t *scaled(cairo_surface_t *src, double kx, double ky, int mirror)
{
    int w = (int)ceil(sw(src) * kx), h = (int)ceil(sh_(src) * ky);
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w < 1 ? 1 : w, h < 1 ? 1 : h);
    cairo_t *cr = cairo_create(s);
    if (mirror) {
        cairo_translate(cr, w, 0);
        cairo_scale(cr, -1, 1);
    }
    cairo_scale(cr, kx, ky);
    cairo_set_source_surface(cr, src, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BEST);
    cairo_paint(cr);
    cairo_destroy(cr);
    return s;
}

static cairo_surface_t *radial(int r, rgb c, double a0, double mid)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 2 * r, 2 * r);
    cairo_t *cr = cairo_create(s);
    cairo_pattern_t *g = cairo_pattern_create_radial(r, r, 0, r, r, r);
    cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, a0);
    cairo_pattern_add_color_stop_rgba(g, mid, c.r, c.g, c.b, a0 * 0.35);
    cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

/* A flame of width w and height h, its base at (0,0) and tip at (0,-h) */
static void flame_path(cairo_t *cr, double w, double h)
{
    double r = w / 2;
    cairo_new_path(cr);
    cairo_move_to(cr, 0, -h);
    cairo_curve_to(cr, r * 0.3, -h * 0.66, r, -h * 0.45, r, -r);
    cairo_arc(cr, 0, -r, r, 0, M_PI);
    cairo_curve_to(cr, -r, -h * 0.45, -r * 0.3, -h * 0.66, 0, -h);
    cairo_close_path(cr);
}

static cairo_surface_t *make_flame(double w, double h)
{
    int iw = (int)ceil(w) + 2, ih = (int)ceil(h) + 2;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, iw, ih);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, iw / 2.0, ih - 1);
    flame_path(cr, w, h);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, -h, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.00, 1.0, 0.30, 0.05, 0.0);
    cairo_pattern_add_color_stop_rgba(g, 0.30, 1.0, 0.42, 0.08, 0.75);
    cairo_pattern_add_color_stop_rgba(g, 0.70, 1.0, 0.70, 0.25, 1.0);
    cairo_pattern_add_color_stop_rgba(g, 1.00, 1.0, 0.85, 0.50, 1.0);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_translate(cr, 0, -h * 0.04);
    flame_path(cr, w * 0.5, h * 0.55);
    g = cairo_pattern_create_linear(0, -h * 0.55, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.0, 1.0, 0.95, 0.70, 0.0);
    cairo_pattern_add_color_stop_rgba(g, 0.5, 1.0, 0.97, 0.80, 0.9);
    cairo_pattern_add_color_stop_rgba(g, 1.0, 1.0, 1.00, 0.92, 0.95);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

/* A glowing rune; squashed onto the floor (and redder) when floor is set */
static cairo_surface_t *make_rune(const char *txt, double size, int floor)
{
    int w = (int)(size * 1.9), h = floor ? (int)(size * 1.1) : w;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_text_extents_t e;
    rgb glow = floor ? (rgb){ 1.0, 0.32, 0.08 } : (rgb){ 1.0, 0.62, 0.20 };
    rgb core = floor ? (rgb){ 1.0, 0.75, 0.45 } : (rgb){ 1.0, 0.95, 0.75 };
    cairo_translate(cr, w / 2.0, h / 2.0);
    if (floor)
        cairo_scale(cr, 1, 0.5);
    cairo_select_font_face(cr, "Noto Sans Runic", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, txt, &e);
    cairo_move_to(cr, -e.width / 2 - e.x_bearing, -e.height / 2 - e.y_bearing);
    cairo_text_path(cr, txt);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    const double lw[3] = { size * 0.55, size * 0.34, size * 0.16 }, la[3] = { 0.10, 0.18, 0.35 };
    for (int i = 0; i < 3; i++) {
        cairo_set_line_width(cr, lw[i]);
        cairo_set_source_rgba(cr, glow.r, glow.g, glow.b, la[i]);
        cairo_stroke_preserve(cr);
    }
    set_rgb(cr, core);
    cairo_fill(cr);
    cairo_destroy(cr);
    return s;
}

/* A mound of coins and gems (the offering heap), base along the bottom edge */
static cairo_surface_t *make_heap(int w, int h)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_save(cr);
    cairo_translate(cr, w / 2.0, h);
    cairo_scale(cr, w / 2.0, h);
    cairo_arc(cr, 0, 0, 1, M_PI, 2 * M_PI);
    cairo_restore(cr);
    cairo_clip_preserve(cr);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, 0, h);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.95, 0.72, 0.30);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.45, 0.26, 0.08);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    srand(11);
    for (int i = 0; i < 150; i++) {
        double x = frand() * w, y = h * (0.05 + 0.95 * frand());
        double gem = frand();
        cairo_save(cr);
        cairo_translate(cr, x, y);
        cairo_scale(cr, 1, 0.6);
        cairo_arc(cr, 0, 0, 1.6 + frand() * 1.2, 0, 2 * M_PI);
        cairo_restore(cr);
        if (gem < 0.06)
            cairo_set_source_rgb(cr, 0.85, 0.12, 0.15);
        else if (gem < 0.10)
            cairo_set_source_rgb(cr, 0.25, 0.55, 0.95);
        else
            cairo_set_source_rgb(cr, 0.75 + frand() * 0.25, 0.55 + frand() * 0.25, 0.18 + frand() * 0.15);
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, 0.3, 0.15, 0.03, 0.6);
        cairo_set_line_width(cr, 0.5);
        cairo_stroke(cr);
    }
    cairo_destroy(cr);
    return s;
}

/* Lift the robes out of the dark floor a little (premultiplied ARGB, so clamp to alpha) */
static void brighten(cairo_surface_t *s, double k)
{
    cairo_surface_flush(s);
    unsigned char *d = cairo_image_surface_get_data(s);
    int stride = cairo_image_surface_get_stride(s);
    for (int y = 0; y < sh_(s); y++)
        for (int x = 0; x < sw(s); x++) {
            unsigned char *p = d + y * stride + x * 4;
            for (int c = 0; c < 3; c++) {
                int v = (int)(p[c] * k);
                p[c] = v > p[3] ? p[3] : v;
            }
        }
    cairo_surface_mark_dirty(s);
}

/* The acolyte in each pose at N_ASC sizes, facing right [0] and left [1] */
static void build_acolytes(void)
{
    for (int p = 0; p < N_POSE; p++) {
        char name[64];
        snprintf(name, sizeof(name), "acolyte_%02d.png", p);
        cairo_surface_t *src = load_asset(name);
        for (int k = 0; k < N_ASC; k++) {
            double sc = ASC_MIN + (ASC_MAX - ASC_MIN) * k / (N_ASC - 1.0);
            aco_spr[p][0][k] = scaled(src, sc, sc, 0);
            aco_spr[p][1][k] = scaled(src, sc, sc, 1);
            brighten(aco_spr[p][0][k], 1.35);
            brighten(aco_spr[p][1][k], 1.35);
        }
        /* undead: the walk frames, desaturated to a pale corpse green */
        if (p == P_WALK0 || p == P_WALK1) {
            for (int m = 0; m < 2; m++) {
                cairo_surface_t *z = scaled(src, ASC_MIN, ASC_MIN, m);
                cairo_t *cr = cairo_create(z);
                cairo_set_operator(cr, CAIRO_OPERATOR_HSL_COLOR);
                cairo_set_source_rgb(cr, 0.50, 0.60, 0.48);
                cairo_mask_surface(cr, z, 0, 0);
                cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
                cairo_set_source_rgba(cr, 0.13, 0.16, 0.12, 1);
                cairo_mask_surface(cr, z, 0, 0);
                cairo_destroy(cr);
                zomb_spr[p - P_WALK0][m] = z;
            }
        }
        cairo_surface_destroy(src);
    }
}

static void build_caches(void)
{
    cairo_surface_t *src = load_asset("temple.png");
    bg_img = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(bg_img);
    cairo_scale(cr, SIZE / (double)sw(src), SIZE / (double)sh_(src));
    cairo_set_source_surface(cr, src, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BEST);
    cairo_paint(cr);
    cairo_identity_matrix(cr);
    /* a candlelit vignette: dark towards the rim of the round screen */
    cairo_pattern_t *v = cairo_pattern_create_radial(240, 270, 150, 240, 250, 250);
    cairo_pattern_add_color_stop_rgba(v, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(v, 1, 0.01, 0.005, 0, 0.55);
    cairo_set_source(cr, v);
    cairo_paint(cr);
    cairo_pattern_destroy(v);
    cairo_destroy(cr);
    cairo_surface_destroy(src);

    mono_src = load_asset("monolith.png");
    mono_img = scaled(mono_src, MONO_K, MONO_K, 0);
    src = load_asset("brazier.png");  brazier_img = scaled(src, 0.42, 0.42, 0); cairo_surface_destroy(src);
    src = load_asset("gong.png");     gong_img = scaled(src, 0.40, 0.40, 0);    cairo_surface_destroy(src);
    src = load_asset("bell.png");     bell_img = scaled(src, 0.36, 0.36, 0);    cairo_surface_destroy(src);
    src = load_asset("censer.png");   censer_img = scaled(src, 0.25, 0.25, 0);  cairo_surface_destroy(src);
    src = load_asset("bowl.png");     bowl_img = scaled(src, 0.37, 0.37, 0);    cairo_surface_destroy(src);
    src = load_asset("candle.png");
    for (int i = 0; i < 3; i++)
        candle_img[i] = scaled(src, 0.10 + 0.025 * i, 0.10 + 0.025 * i, 0);
    cairo_surface_destroy(src);
    for (int i = 0; i < 3; i++) {
        char name[32];
        snprintf(name, sizeof(name), "pigeon_%d.png", i);
        pigeon_src[i] = load_asset(name);
    }
    src = load_asset("tablet.png");
    tablet_img = scaled(src, TAB_W / (double)sw(src), TAB_H / (double)sh_(src), 0);
    cairo_surface_destroy(src);
    build_acolytes();

    for (int i = 0; i < N_GLYPH_SPR; i++) {
        glyph_spr[i] = make_rune(RUNES[i], 15, 0);
        rune_spr[i] = make_rune(RUNES[i], 17, 1);
    }
    for (int i = 0; i < N_FLAME; i++) {
        flame_h[i] = 3 + i * 1.0;
        flame_spr[i] = make_flame(2.4 + flame_h[i] * 0.3, flame_h[i]);
    }
    big_flame = make_flame(40, 120);
    for (int i = 0; i < N_PUFF_SPR; i++) {
        int r = 4 + i * 2;
        puff_spr[i] = radial(r, (rgb){ 0.62, 0.56, 0.62 }, 0.55, 0.5);
    }
    shadow_spr = radial(20, (rgb){ 0, 0, 0 }, 0.9, 0.55);
    spark_spr = radial(3, (rgb){ 1.0, 0.75, 0.35 }, 1.0, 0.6);
    src = radial(130, (rgb){ 1.0, 0.78, 0.45 }, 0.55, 0.35);
    halo_cool = scaled(src, 0.95, 1.25, 0);
    cairo_surface_destroy(src);
    src = radial(130, (rgb){ 1.0, 0.12, 0.05 }, 0.75, 0.35);
    halo_hot = scaled(src, 1.05, 1.35, 0);
    cairo_surface_destroy(src);
    fire_glow = radial(95, (rgb){ 1.0, 0.45, 0.12 }, 0.45, 0.3);
    candle_glow = radial(14, (rgb){ 1.0, 0.55, 0.2 }, 0.5, 0.3);
    heap_glow = radial(40, (rgb){ 1.0, 0.7, 0.25 }, 0.4, 0.4);
    heap_img = make_heap(50, 30);

    zzz_spr = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 16, 18);
    cr = cairo_create(zzz_spr);
    cairo_select_font_face(cr, "DejaVu Serif", CAIRO_FONT_SLANT_ITALIC, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 14);
    cairo_move_to(cr, 3, 14);
    cairo_text_path(cr, "z");
    cairo_set_source_rgba(cr, 0, 0, 0, 0.6);
    cairo_set_line_width(cr, 2.5);
    cairo_stroke_preserve(cr);
    cairo_set_source_rgb(cr, 0.85, 0.82, 0.95);
    cairo_fill(cr);
    cairo_destroy(cr);

    /* candles: a ring just inside the carved rim, leaving a gap behind the tablet */
    for (int i = 0; i < N_CANDLES; i++) {
        double gap = 44 * M_PI / 180;                       /* centred on straight down */
        double a = M_PI / 2 + gap / 2 + (2 * M_PI - gap) * (i + 0.5) / N_CANDLES;
        cand_x[i] = RING_CX + RING_RX * cos(a);
        cand_y[i] = RING_CY + RING_RY * sin(a);
    }
    /* floor runes on the carved rim, where they're on screen and not under the tablet */
    int n = 0;
    srand(3);
    for (int i = 0; n < N_RUNES && i < 200; i++) {
        double a = 2 * M_PI * i / 34.0 + 0.05;
        double x = 240 + 214 * cos(a), y = 292 + 101 * sin(a);
        if (hypot(x - 240, y - 240) > 222 || (y > 355 && fabs(x - 240) < 120) || (y < 250 && fabs(x - 240) < 80))
            continue;
        rune_x[n] = x, rune_y[n] = y, rune_g[n] = rand() % N_GLYPH_SPR;
        n++;
    }
    for (; n < N_RUNES; n++)
        rune_x[n] = -100, rune_y[n] = -100;
    srand((unsigned)time(NULL));

    /* acolyte places, filled in this order: front, sides, back, front-most */
    static const double ang[4] = { 30, -5, -40, 65 };
    for (int i = 0; i < MAX_ACO; i++) {
        double a = ang[i / 2] * M_PI / 180;
        int left = i & 1;
        slot_x[i] = 240 + (left ? -1 : 1) * 158 * cos(a);
        slot_y[i] = 297 + 64 * sin(a);
        aco[i].off = frand();
        aco[i].slot = i;
    }

    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, TAB_W, TAB_H);
    cap_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, CAP_W, CAP_H);
}

/* ---------------------------------------------------------------- simulation */

static double depth_k(double y) { return ASC_MIN + (ASC_MAX - ASC_MIN) * clamp01((y - 222) / (356 - 222)); }

static void set_frame(acolyte *a, int f, double fdur)
{
    if (f == a->frame)
        return;
    a->prev = a->frame;
    a->frame = f;
    a->fade = fdur > 0 ? 0 : 1;
    a->fdur = fdur;
}

/* Walk towards (tx,ty); faces the way it's going (only when clearly moving sideways) */
static int walk_to(acolyte *a, double tx, double ty, double speed, double dt)
{
    double dx = tx - a->x, dy = ty - a->y, d = hypot(dx, dy), step = speed * dt;
    if (d <= step || d < 0.5) {
        a->x = tx, a->y = ty;
        return 1;
    }
    a->x += dx / d * step;
    a->y += dy / d * step;
    if (fabs(dx) > 0.35 * d)
        a->face = dx > 0 ? 1 : -1;
    a->walk += step;
    set_frame(a, ((int)(a->walk / 9) & 1) ? P_WALK1 : P_WALK0, 0);
    return 0;
}

static double edge_x(double y, int left)
{
    double dy = y - 240, h = sqrt(fmax(0, 240.0 * 240.0 - dy * dy));
    return left ? 240 - h - 30 : 240 + h + 30;
}

/* The pose for a point in the chant; frenzy swaps the upright prayer for arms in the air */
static int chant_pose(double p, int frenzy)
{
    p -= floor(p);
    if (p < (frenzy ? 0.40 : 0.50))
        return frenzy ? P_RAISE : P_UP;
    if (p < (frenzy ? 0.50 : 0.60))
        return P_HALF;
    if (p < 0.84)
        return P_BOW;
    return P_HALF;
}

static void update_acolyte(acolyte *a, int i, double dt, double t)
{
    int want = i < W.count, left = slot_x[i] < 240;
    int slot_face = left ? 1 : -1;
    double tempo = 0.16 + 1.15 * W.act;
    double fd = fmin(0.28, fmax(0.07, 0.10 / tempo));

    switch (a->state) {
    case A_OFF:
        if (want) {
            a->state = A_ENTER;
            a->x = edge_x(slot_y[i], left);
            a->y = slot_y[i];
            a->face = slot_face;
            a->frame = a->prev = P_WALK0;
            a->fade = 1;
        }
        break;
    case A_ENTER:
    case A_RETURN:
        if (!want) {
            a->state = A_LEAVE;
        } else if (walk_to(a, slot_x[i], slot_y[i], WALK_SPEED, dt)) {
            a->face = slot_face;
            a->state = W.idle ? A_SLEEP : A_KNEEL;
            set_frame(a, a->state == A_SLEEP ? (i & 1 ? P_CURL : P_SIT) : P_UP, 0.25);
        }
        break;
    case A_KNEEL: {
        double off = a->off * 0.4 * W.frenzy_x;             /* in step when calm, ragged in a frenzy */
        set_frame(a, chant_pose(W.chant_ph + off, W.frenzy), fd);
        if (!want)
            a->state = A_LEAVE;
        else if (W.idle)
            a->state = i == 0 ? A_TO_SWEEP : A_SLEEP;
        break;
    }
    case A_SLEEP:
        set_frame(a, i & 1 ? P_CURL : P_SIT, 0.5);
        if (!want)
            a->state = A_LEAVE;
        else if (!W.idle)
            a->state = A_KNEEL;
        break;
    case A_TO_SWEEP:
        if (!W.idle)
            a->state = A_RETURN;
        else if (walk_to(a, a->x > 240 ? SWEEP_X1 : SWEEP_X0, SWEEP_Y, WALK_SPEED, dt)) {
            a->state = A_SWEEP;
            a->dir = a->x > 240 ? -1 : 1;
        }
        break;
    case A_SWEEP: {
        if (!W.idle) {
            a->state = A_RETURN;
            break;
        }
        a->x += a->dir * 14 * dt;
        if (a->x < SWEEP_X0)
            a->dir = 1;
        else if (a->x > SWEEP_X1)
            a->dir = -1;
        a->face = a->dir > 0 ? 1 : -1;
        set_frame(a, fmod(t, 0.9) < 0.45 ? P_SWEEP0 : P_SWEEP1, 0.15);
        break;
    }
    case A_LEAVE:
        if (want) {
            a->state = A_ENTER;
        } else if (walk_to(a, edge_x(a->y, left), a->y, WALK_SPEED, dt)) {
            a->state = A_OFF;
        }
        break;
    }
    a->fade = fmin(1, a->fade + (a->fdur > 0 ? dt / a->fdur : 1));
}

static void update_fanner(double dt, double t)
{
    acolyte *a = &fanner;
    switch (a->state) {
    case A_OFF:
        if (W.hot) {
            a->state = A_ENTER;
            a->x = edge_x(FAN_Y, 0);
            a->y = FAN_Y;
            a->face = -1;
            a->frame = a->prev = P_WALK0;
            a->fade = 1;
        }
        break;
    case A_ENTER:
        if (!W.hot)
            a->state = A_LEAVE;
        else if (walk_to(a, FAN_X, FAN_Y, WALK_SPEED * 1.4, dt)) {   /* hurries */
            a->face = -1;
            a->state = A_KNEEL;
        }
        break;
    case A_KNEEL:
        set_frame(a, fmod(t, 0.56) < 0.28 ? P_FAN0 : P_FAN1, 0.1);
        if (!W.hot)
            a->state = A_LEAVE;
        break;
    default:
        if (W.hot)
            a->state = A_ENTER;
        else if (walk_to(a, edge_x(FAN_Y, 0), FAN_Y, WALK_SPEED, dt))
            a->state = A_OFF;
        break;
    }
    a->fade = fmin(1, a->fade + (a->fdur > 0 ? dt / a->fdur : 1));
}

/* Zombie processes: undead acolytes shambling to and fro behind the circle */
static void update_zombie(acolyte *a, int j, double dt)
{
    int want = j < W.zomb_n, left = j & 1;
    double y = ZOMB_Y + j * 5;
    switch (a->state) {
    case A_OFF:
        if (want) {
            a->state = A_ENTER;
            a->x = edge_x(y, left);
            a->y = y;
            a->dir = left ? 1 : -1;
            a->face = (int)a->dir;
        }
        break;
    case A_ENTER:
        if (!want) {
            a->state = A_LEAVE;
            break;
        }
        {
            double v = (a->x < 130 || a->x > 350) ? 24 : 9;     /* hurry in, then shamble */
            a->x += a->dir * v * dt;
            a->walk += v * dt;
        }
        if (a->x < 140)
            a->dir = 1;
        else if (a->x > 340)
            a->dir = -1;
        a->face = a->dir > 0 ? 1 : -1;
        break;
    case A_LEAVE: {
        if (want) {
            a->state = A_ENTER;
            break;
        }
        double tx = edge_x(y, a->x < 240);
        a->dir = tx > a->x ? 1 : -1;
        a->face = (int)a->dir;
        a->x += a->dir * 14 * dt;
        a->walk += 14 * dt;
        if ((a->dir > 0 && a->x >= tx) || (a->dir < 0 && a->x <= tx))
            a->state = A_OFF;
        break;
    }
    }
    a->sway = sin(a->walk * 0.35) * 0.07;
}

static void spawn_glyph(void)
{
    int k[MAX_ACO], n = 0;
    for (int i = 0; i < MAX_ACO; i++)
        if (aco[i].state == A_KNEEL)
            k[n++] = i;
    if (!n)
        return;
    const acolyte *a = &aco[k[rand() % n]];
    for (int i = 0; i < MAX_GLYPH; i++) {
        glyph *g = &glyphs[i];
        if (g->alive)
            continue;
        double kk = depth_k(a->y);
        g->x0 = a->x + a->face * 8 * kk / 0.4;
        g->y0 = a->y - 95 * kk;
        g->x1 = MONO_X + (frand() - 0.5) * 30;
        g->y1 = 118 + frand() * 110;
        g->cx = (g->x0 + g->x1) / 2 + (frand() - 0.5) * 40;
        g->cy = fmin(g->y0, g->y1) - 30 - frand() * 50;
        g->t = 0;
        g->dur = 1.4 + frand() * 0.9;
        g->g = rand() % N_GLYPH_SPR;
        g->alive = 1;
        return;
    }
}

static void add_ring(ring *r)
{
    for (int i = 0; i < MAX_RING; i++)
        if (!r[i].alive) {
            r[i].alive = 1;
            r[i].age = 0;
            return;
        }
}

/* log-scale a rate between lo (0) and hi (1) */
static double logk(double v, double lo, double hi) { return v <= lo ? 0 : clamp01(log(v / lo) / log(hi / lo)); }

static void simulate(const stats *g, const sys_stats *s, double shown_tok, double dt, double t)
{
    double k1 = fmin(1, dt * 1.5), k2 = fmin(1, dt * 0.6);

    /* chant intensity from tok/s (or GPU activity), machine load for the head count */
    double act = gpu_source ? shown_rate(shown_tok) / 100 : clamp01(shown_tok / 1500);
    double gpu_act = (gpu_activity(g, 0) + gpu_activity(g, 1)) / 2;
    double disk = logk(s->disk_rd + s->disk_wr, 1e5, 3e9), net_rx = logk(s->net_rx, 2e3, 1e8), net_tx = logk(s->net_tx, 2e3, 1e8);
    double mload = clamp01(0.45 * gpu_act + 0.35 * fmin(1, s->cpu_load * 1.4) + 0.1 * disk + 0.1 * fmax(net_rx, net_tx));
    double heat_gpu = clamp01((fmax(g->temp[0], g->temp[1]) - 50) / (84.0 - 50));
    double heat_cpu = clamp01((s->cpu_temp - 55) / (92.0 - 55));
    W.act += (act - W.act) * k1;
    W.mload += (mload - W.mload) * k2;
    W.heat += (fmax(heat_gpu, heat_cpu) - W.heat) * fmin(1, dt * 1.2);
    for (int i = 0; i < 2; i++)
        W.pw[i] += (clamp01((g->power[i] - 25) / (GPU_MAX_W - 25)) - W.pw[i]) * fmin(1, dt * 2.5);
    W.fanfrac += ((g->fan[0] + g->fan[1]) / 2 - W.fanfrac) * k2;
    W.ram += ((s->ram_total > 0 ? s->ram_used / s->ram_total : 0) - W.ram) * k2;
    W.disk += (disk - W.disk) * k1;
    W.net_rx += (net_rx - W.net_rx) * k1;
    W.net_tx += (net_tx - W.net_tx) * k1;
    W.intr += (logk(s->intr_s, 2e4, 1.5e6) - W.intr) * k2;
    W.forks += (s->forks_s - W.forks) * k1;
    W.entropy += (clamp01(s->entropy / 256) - W.entropy) * k2;
    for (int i = 0; i < N_CANDLES; i++)
        cand_l[i] += ((i < s->n_threads ? s->thread_load[i] : 0) - cand_l[i]) * fmin(1, dt * 3);

    /* moods, all with hysteresis so nothing flickers */
    int idle_raw = act < 0.02 && g->running == 0 && s->cpu_load < 0.12;
    W.idle_t = idle_raw != W.idle ? W.idle_t + dt : 0;
    if (W.idle_t > 2.5) {
        W.idle = idle_raw;
        W.idle_t = 0;
    }
    if (!W.frenzy && W.act > 0.75)
        W.frenzy = 1;
    else if (W.frenzy && W.act < 0.62)
        W.frenzy = 0;
    W.frenzy_x += ((W.frenzy ? 1 : 0) - W.frenzy_x) * fmin(1, dt * 0.8);
    int hot_raw = W.hot ? W.heat > 0.8 : W.heat > 0.92;
    W.hot_t = hot_raw != W.hot ? W.hot_t + dt : 0;
    if (W.hot_t > 1.0)
        W.hot = hot_raw, W.hot_t = 0;
    int want = ACO_MIN + (int)floor(W.mload * (MAX_ACO - ACO_MIN) + 0.5);
    if (want != W.count && abs(want - W.count) >= 1 && fabs(W.mload * (MAX_ACO - ACO_MIN) - (W.count - ACO_MIN)) > 0.62)
        W.count_t += dt;
    else
        W.count_t = 0;
    if (W.count_t > 1.2) {                                   /* one at a time */
        W.count += want > W.count ? 1 : -1;
        W.count_t = 0;
    }
    W.zomb_n = s->zombies > MAX_ZOMB ? MAX_ZOMB : s->zombies;
    {
        static const double up[4] = { 0.2, 0.45, 0.75, 9 };
        int m = W.mood;
        if (m < 3 && W.act > up[m] + 0.03)
            m++;
        else if (m > 0 && W.act < up[m - 1] - 0.03)
            m--;
        W.mood = m;
    }

    W.chant_ph += (0.16 + 1.15 * W.act) * dt;
    W.fan_ang += (0.5 + 5.5 * W.fanfrac) * dt;
    W.pulse *= exp(-dt * 2.2);

    for (int i = 0; i < MAX_ACO; i++)
        update_acolyte(&aco[i], i, dt, t);
    update_fanner(dt, t);
    for (int j = 0; j < MAX_ZOMB; j++)
        update_zombie(&zomb[j], j, dt);

    /* offerings: runes rising into the monolith, one per ~40 tokens */
    W.spawn_glyph += fmin(12, (gpu_source ? act * GPU_FULL_RATE * N_GPUS : shown_tok) / 40) * dt;
    while (W.spawn_glyph >= 1) {
        spawn_glyph();
        W.spawn_glyph -= 1;
    }
    for (int i = 0; i < MAX_GLYPH; i++) {
        glyph *gl = &glyphs[i];
        if (!gl->alive)
            continue;
        gl->t += dt / gl->dur;
        if (gl->t >= 1) {
            gl->alive = 0;
            W.pulse = fmin(1, W.pulse + 0.08);
        }
    }

    /* forks light runes on the floor */
    W.spawn_fork += fmin(14, W.forks * 0.6) * dt;
    while (W.spawn_fork >= 1) {
        int r = rand() % N_RUNES;
        if (rune_x[r] > 0)
            rune_lit[r] = 1;
        W.spawn_fork -= 1;
    }
    for (int i = 0; i < N_RUNES; i++)
        rune_lit[i] *= exp(-dt * 1.6);

    /* incense (entropy) */
    for (int c = 0; c < 2; c++) {
        W.spawn_puff[c] += (0.6 + 3.2 * W.entropy) * dt;
        while (W.spawn_puff[c] >= 1) {
            W.spawn_puff[c] -= 1;
            for (int i = 0; i < MAX_PUFF; i++)
                if (!puffs[i].alive) {
                    puff *p = &puffs[i];
                    p->x = CENSER_X[c] + (frand() - 0.5) * 3;
                    p->y = CENSER_Y - 27;
                    p->vx = (frand() - 0.5) * 5;
                    p->age = 0;
                    p->life = 3.5 + frand() * 1.5;
                    p->size = 0.9 + frand() * 0.3;
                    p->alive = 1;
                    break;
                }
        }
    }
    for (int i = 0; i < MAX_PUFF; i++) {
        puff *p = &puffs[i];
        if (!p->alive)
            continue;
        p->age += dt;
        p->y -= (13 - 2 * p->age) * dt;
        p->x += (p->vx + 5 * sin(p->age * 1.3 + i)) * dt;
        if (p->age > p->life)
            p->alive = 0;
    }

    /* sparks off the braziers at high power */
    for (int b = 0; b < 2; b++) {
        W.spawn_spark[b] += 14 * W.pw[b] * W.pw[b] * dt;
        while (W.spawn_spark[b] >= 1) {
            W.spawn_spark[b] -= 1;
            for (int i = 0; i < MAX_SPARK; i++)
                if (!sparks[i].alive) {
                    spark *p = &sparks[i];
                    p->x = BRZ_X[b] + (frand() - 0.5) * 20;
                    p->y = BRZ_Y - 50;
                    p->vx = (frand() - 0.5) * 22;
                    p->vy = -(40 + 70 * frand()) * (0.5 + W.pw[b]);
                    p->age = 0;
                    p->life = 0.8 + frand() * 0.8;
                    p->alive = 1;
                    break;
                }
        }
    }
    for (int i = 0; i < MAX_SPARK; i++) {
        spark *p = &sparks[i];
        if (!p->alive)
            continue;
        p->age += dt;
        p->x += (p->vx + 10 * sin(t * 3 + i)) * dt;
        p->y += p->vy * dt;
        p->vy *= exp(-dt * 0.8);
        if (p->age > p->life || hypot(p->x - 240, p->y - 240) > 228)
            p->alive = 0;
    }

    /* the gong: NVMe traffic */
    if (W.disk > 0.02 && t >= W.gong_next) {
        W.gong_next = t + 3.2 - 2.6 * W.disk;
        W.gong_flash = 0.4 + 0.6 * W.disk;
        W.gong_shake = 1;
        add_ring(gong_rings);
    } else if (W.disk <= 0.02) {
        W.gong_next = fmin(W.gong_next, t + 0.5);
    }
    W.gong_flash *= exp(-dt * 3);
    W.gong_shake *= exp(-dt * 5);
    for (int i = 0; i < MAX_RING; i++) {
        if (gong_rings[i].alive && (gong_rings[i].age += dt) > 1.4)
            gong_rings[i].alive = 0;
        if (bell_rings[i].alive && (bell_rings[i].age += dt) > 0.9)
            bell_rings[i].alive = 0;
    }

    /* the prayer bell: interrupts; chimes at each end of its swing */
    W.bell_amp += (0.04 + 0.30 * W.intr - W.bell_amp) * k2;
    double ph0 = W.bell_ph;
    W.bell_ph += dt * 4.2;
    if (floor(ph0 / M_PI + 0.5) != floor(W.bell_ph / M_PI + 0.5) && W.bell_amp > 0.08)
        add_ring(bell_rings);

    /* carrier pigeons: network in (from the left, into the monolith) and out (to the right) */
    for (int d = 0; d < 2; d++) {
        double r = d ? W.net_tx : W.net_rx;
        W.spawn_pig[d] += (r > 0.03 ? 0.1 + 0.8 * r : 0) * dt;
        while (W.spawn_pig[d] >= 1) {
            W.spawn_pig[d] -= 1;
            for (int i = 0; i < MAX_PIGEON; i++)
                if (!pigeons[i].alive) {
                    pigeon *p = &pigeons[i];
                    p->alive = 1;
                    p->out = d;
                    p->t = 0;
                    p->dur = 3.4 + frand() * 1.2;
                    p->y0 = 60 + frand() * 60;
                    p->y1 = 50 + frand() * 70;
                    break;
                }
        }
    }
    for (int i = 0; i < MAX_PIGEON; i++)
        if (pigeons[i].alive && (pigeons[i].t += dt / pigeons[i].dur) >= 1)
            pigeons[i].alive = 0;
}

/* ---------------------------------------------------------------- render */

/* Whole-pixel positions keep pixman on its fast untransformed path */
static void blit(cairo_t *cr, cairo_surface_t *s, double x, double y, double a)
{
    cairo_set_source_surface(cr, s, floor(x + 0.5), floor(y + 0.5));
    if (a >= 0.999)
        cairo_paint(cr);
    else if (a > 0.004)
        cairo_paint_with_alpha(cr, a);
}

/* Blit centred on (x,y) scaled by (kx,ky) */
static void blit_scaled(cairo_t *cr, cairo_surface_t *s, double x, double y, double kx, double ky, double a)
{
    double w = sw(s), h = sh_(s);
    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, kx, ky);
    cairo_rectangle(cr, -w / 2, -h / 2, w, h);
    cairo_clip(cr);
    cairo_set_source_surface(cr, s, -w / 2, -h / 2);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint_with_alpha(cr, a);
    cairo_restore(cr);
}

static void draw_pose(cairo_t *cr, int pose, int face, double x, double y, double a)
{
    double kk = depth_k(y);
    int si = (int)lround((kk - ASC_MIN) / (ASC_MAX - ASC_MIN) * (N_ASC - 1));
    si = si < 0 ? 0 : si >= N_ASC ? N_ASC - 1 : si;
    kk = ASC_MIN + (ASC_MAX - ASC_MIN) * si / (N_ASC - 1.0);
    cairo_surface_t *s = aco_spr[pose][face < 0][si];
    double w = sw(s), h = sh_(s);
    double ax = POSE[pose].ax * kk;
    double anchor = POSE[pose].kneel ? x - face * KNEEL_HALF * kk : x;     /* back of the robe, or the feet */
    double left = face > 0 ? anchor - ax : anchor - (w - ax);
    if (pose != P_CURL && a > 0.5) {
        double sww = (POSE[pose].kneel ? 70 : 44) * kk / 0.4;
        blit_scaled(cr, shadow_spr, x, y - 1, sww / 40.0, 0.22 * sww / 40.0, 0.55);
    }
    blit(cr, s, left, y - h + 2, a);
}

static void draw_acolyte(cairo_t *cr, const acolyte *a)
{
    if (a->fade < 1)
        draw_pose(cr, a->prev, a->face, a->x, a->y, 1 - a->fade * a->fade);
    draw_pose(cr, a->frame, a->face, a->x, a->y, fmin(1, a->fade * 1.5 + 0.001));
}

static void draw_zombie(cairo_t *cr, const acolyte *a)
{
    cairo_surface_t *s = zomb_spr[((int)(a->walk / 7)) & 1][a->face < 0];
    double w = sw(s), h = sh_(s);
    cairo_save(cr);
    cairo_translate(cr, a->x, a->y);
    cairo_rotate(cr, a->sway);
    blit(cr, s, -w / 2, -h + 2, 0.8);
    cairo_restore(cr);
}

static void draw_candle(cairo_t *cr, int i, double t)
{
    double x = cand_x[i], y = cand_y[i];
    double d = clamp01((y - (RING_CY - RING_RY)) / (2 * RING_RY));
    cairo_surface_t *c = candle_img[d < 0.33 ? 0 : d < 0.66 ? 1 : 2];
    double k = 0.75 + 0.4 * d, l = cand_l[i];
    double w = sw(c), h = sh_(c);
    blit(cr, c, x - w / 2, y - h, 1);
    double fh = (3 + 13 * l) * k * (1 + 0.1 * sin(t * (9 + i % 5) + i * 2.3));
    int fi = (int)lround(fh - 3);
    fi = fi < 0 ? 0 : fi >= N_FLAME ? N_FLAME - 1 : fi;
    cairo_surface_t *f = flame_spr[fi];
    double wick = y - h + 1;
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    blit(cr, candle_glow, x - 14, wick - 4 - 14, 0.25 + 0.6 * l);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    blit(cr, f, x - sw(f) / 2.0 + 0.4 * sin(t * 5 + i), wick - sh_(f) + 1, 1);
}

static void draw_brazier(cairo_t *cr, int b, double t)
{
    double x = BRZ_X[b], p = W.pw[b];
    double w = sw(brazier_img), h = sh_(brazier_img);
    blit(cr, brazier_img, x - w / 2, BRZ_Y - h, 1);
    double base = BRZ_Y - h + 9, fh = 14 + 96 * p, fw = 18 + 16 * p;
    for (int k = 0; k < 3; k++) {
        int j = k == 2 ? 1 : k == 1 ? 2 : 0;                    /* side tongues first, centre on top */
        double hk = fh * (j == 1 ? 1 : 0.6) * (1 + 0.12 * sin(t * (7 + j * 1.3) + j * 2 + b));
        double wk = fw * (j == 1 ? 1 : 0.62);
        double ox = (j - 1) * fw * 0.3 + sin(t * 3.1 + j + b * 2) * (1 + 2 * p);
        cairo_save(cr);
        cairo_translate(cr, x + ox, base + 3);
        cairo_scale(cr, wk / 42.0, hk / 122.0);
        cairo_set_source_surface(cr, big_flame, -21, -121);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
        cairo_rectangle(cr, -21, -121, 42, 122);
        cairo_clip(cr);
        cairo_paint_with_alpha(cr, 0.92);
        cairo_restore(cr);
    }
}

static void draw_gong(cairo_t *cr, double t)
{
    double w = sw(gong_img), h = sh_(gong_img);
    double sx = W.gong_shake * sin(t * 40) * 1.5;
    double x = GONG_X - w / 2 + sx, y = GONG_Y - h, cx = x + 31, cy = y + 28;
    blit(cr, gong_img, x, y, 1);
    if (W.gong_flash > 0.02) {
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        cairo_set_source_rgba(cr, 1.0, 0.7, 0.3, 0.45 * W.gong_flash);
        cairo_arc(cr, cx, cy, 21, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    }
    cairo_set_line_width(cr, 1.6);
    for (int i = 0; i < MAX_RING; i++) {
        if (!gong_rings[i].alive)
            continue;
        double a = gong_rings[i].age, r = 22 + 55 * a;
        cairo_save(cr);
        cairo_translate(cr, cx, cy);
        cairo_scale(cr, 1, 0.85);
        cairo_arc(cr, 0, 0, r, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, 1.0, 0.78, 0.4, 0.55 * (1 - a / 1.4) * (1 - a / 1.4));
        cairo_stroke(cr);
    }
}

static void draw_bell(cairo_t *cr, double t)
{
    (void)t;
    double w = sw(bell_img), h = sh_(bell_img);
    const double hx = 25, hy = 14;                  /* the hinge on the bracket */
    double ang = W.bell_amp * sin(W.bell_ph);
    /* bracket (static) */
    cairo_save(cr);
    cairo_rectangle(cr, BELL_X, BELL_Y, 11, h);
    cairo_rectangle(cr, BELL_X + 11, BELL_Y, w - 11, hy);
    cairo_clip(cr);
    blit(cr, bell_img, BELL_X, BELL_Y, 1);
    cairo_restore(cr);
    /* bell on its chain, swinging */
    cairo_save(cr);
    cairo_translate(cr, BELL_X + hx, BELL_Y + hy);
    cairo_rotate(cr, ang);
    cairo_rectangle(cr, 11 - hx, 0, w - 11, h - hy);
    cairo_clip(cr);
    cairo_set_source_surface(cr, bell_img, -hx, -hy);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);
    cairo_restore(cr);
    cairo_set_line_width(cr, 1.3);
    for (int i = 0; i < MAX_RING; i++) {
        if (!bell_rings[i].alive)
            continue;
        double a = bell_rings[i].age, r = 10 + 30 * a;
        cairo_save(cr);
        cairo_translate(cr, BELL_X + hx, BELL_Y + 40);
        cairo_scale(cr, 1, 0.8);
        cairo_arc(cr, 0, 0, r, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, 1.0, 0.85, 0.5, 0.45 * (1 - a / 0.9));
        cairo_stroke(cr);
    }
}

/* fan hub centres on the monolith asset, and their radius (asset px) */
static const double FAN_CX = 140.4, FAN_CY[3] = { 71.1, 177.1, 282.5 }, FAN_R = 41.0;

static void draw_monolith(cairo_t *cr, double t)
{
    double w = sw(mono_img), h = sh_(mono_img);
    double x = MONO_X - w / 2, y = MONO_BASE - h;
    blit(cr, mono_img, x, y, 1);
    /* spinning fans */
    for (int i = 0; i < 3; i++) {
        cairo_save(cr);
        cairo_translate(cr, x + FAN_CX * MONO_K, y + FAN_CY[i] * MONO_K);
        cairo_arc(cr, 0, 0, FAN_R * MONO_K, 0, 2 * M_PI);
        cairo_clip(cr);
        cairo_scale(cr, MONO_K, MONO_K);
        cairo_rotate(cr, W.fan_ang + i * 0.9);
        cairo_set_source_surface(cr, mono_src, -FAN_CX, -FAN_CY[i]);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
        cairo_paint(cr);
        cairo_restore(cr);
    }
    /* holy (or overheated) light on the card */
    double a = 0.03 + 0.07 * W.act + 0.08 * W.pulse;
    rgb c = lerp((rgb){ 1.0, 0.72, 0.38 }, (rgb){ 1.0, 0.10, 0.04 }, W.heat * W.heat);
    a += 0.22 * W.heat * W.heat * (0.8 + 0.2 * sin(t * 5));
    cairo_save(cr);
    cairo_rectangle(cr, x + 40, y, w - 80, 170);
    cairo_clip(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_set_source_rgba(cr, c.r, c.g, c.b, a);
    cairo_mask_surface(cr, mono_img, x, y);
    cairo_restore(cr);
    /* the altar's candles */
    for (int i = 0; i < 2; i++) {
        cairo_surface_t *f = flame_spr[7];
        double fx = x + (i ? 240.0 : 43.7) * MONO_K, fy = y + 307 * MONO_K;
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        blit(cr, candle_glow, fx - 14, fy - 18, 0.7);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
        blit(cr, f, fx - sw(f) / 2.0 + 0.4 * sin(t * 6 + i), fy - sh_(f) + 1, 1);
    }
}

static void draw_bowl(cairo_t *cr, double t)
{
    (void)t;
    double w = sw(bowl_img), h = sh_(bowl_img);
    double x = 240 - w / 2, y = BOWL_Y - h, rim = y + 6;
    blit(cr, bowl_img, x, y, 1);
    double hh = 2 + 26 * W.ram;
    cairo_save(cr);
    cairo_rectangle(cr, x, rim - hh - 2, w, hh + 3);
    cairo_clip(cr);
    cairo_translate(cr, 240, rim + 1);
    cairo_scale(cr, 0.96 * (w * 0.9) / 50.0, hh / 30.0);
    cairo_set_source_surface(cr, heap_img, -25, -30);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);
    cairo_restore(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    blit(cr, heap_glow, 240 - 40, rim - hh * 0.5 - 40, 0.15 + 0.35 * W.ram);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

typedef struct { double y; int kind, idx; } item;
enum { K_CANDLE, K_ACO, K_FAN, K_ZOMB, K_BRAZ, K_GONG, K_BELL, K_MONO, K_CENSER, K_BOWL };

static int by_y(const void *a, const void *b)
{
    double ya = ((const item *)a)->y, yb = ((const item *)b)->y;
    return ya < yb ? -1 : ya > yb ? 1 : 0;
}

static void fmt_bytes(char *out, size_t n, double b)
{
    if (b >= 1e9)
        snprintf(out, n, "%.1fG", b / 1e9);
    else if (b >= 1e6)
        snprintf(out, n, "%.0fM", b / 1e6);
    else if (b >= 1e3)
        snprintf(out, n, "%.0fK", b / 1e3);
    else
        snprintf(out, n, "%.0f", b);
}

/*
 * The tablet: the big number (tok/s or % GPU) and a line that cycles every 3 s
 * through the rest of the machine. Redrawn only when its text changes (<= 4x/s).
 */
static void update_hud(const stats *g, const sys_stats *s, double shown_tok, double t)
{
    static char key[160];
    static double next, tok;
    char k[160], big[32], lab[48], val[48], b1[16], b2[16];
    if (t < next && t > next - 1)
        return;
    next = t + 0.25;
    tok = shown_rate(shown_tok);

    int item = (int)(t / 3.0) % 10;
    if (item == 8 && s->zombies == 0)
        item = 9;
    switch (item) {
    case 0: snprintf(lab, sizeof(lab), "GPU "); snprintf(val, sizeof(val), "%.0f W \xc2\xb7 %d\xc2\xb0", g->power[0] + g->power[1], g->temp[0] > g->temp[1] ? g->temp[0] : g->temp[1]); break;
    case 1: snprintf(lab, sizeof(lab), "CPU "); snprintf(val, sizeof(val), "%.0f%% \xc2\xb7 %.0f\xc2\xb0", s->cpu_load * 100, s->cpu_temp); break;
    case 2: snprintf(lab, sizeof(lab), "RAM "); snprintf(val, sizeof(val), "%.0f / %.0f GB", s->ram_used, s->ram_total); break;
    case 3: fmt_bytes(b1, sizeof(b1), s->disk_rd + s->disk_wr);
            snprintf(lab, sizeof(lab), "GONG "); snprintf(val, sizeof(val), "%sB/s", b1); break;
    case 4: fmt_bytes(b1, sizeof(b1), s->net_rx); fmt_bytes(b2, sizeof(b2), s->net_tx);
            snprintf(lab, sizeof(lab), "DOVES "); snprintf(val, sizeof(val), "\xe2\x86\x93%s \xe2\x86\x91%s", b1, b2); break;
    case 5: snprintf(lab, sizeof(lab), "BELLS "); snprintf(val, sizeof(val), "%.0fk/s", s->intr_s / 1000); break;
    case 6: snprintf(lab, sizeof(lab), "SOULS "); snprintf(val, sizeof(val), "%d", s->tasks); break;
    case 7: snprintf(lab, sizeof(lab), "INITIATES "); snprintf(val, sizeof(val), "%.0f/s", s->forks_s); break;
    case 8: snprintf(lab, sizeof(lab), "UNDEAD "); snprintf(val, sizeof(val), "%d", s->zombies); break;
    default:
        snprintf(lab, sizeof(lab), "VIGIL ");
        if (s->uptime >= 2 * 86400)
            snprintf(val, sizeof(val), "%.0f DAYS", floor(s->uptime / 86400));
        else
            snprintf(val, sizeof(val), "%.0f H", floor(s->uptime / 3600));
        break;
    }
    snprintf(big, sizeof(big), "%.0f", tok);
    snprintf(k, sizeof(k), "%s|%s|%s", big, lab, val);
    if (!strcmp(k, key))
        return;
    strcpy(key, k);

    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
    cairo_set_source_surface(cr, tablet_img, 0, 0);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* big number + unit, fitted to the face */
    const char *unit = gpu_source ? "% GPU" : " tok/s";
    cairo_text_extents_t e1, e2;
    double s1 = 34, s2 = 22, maxw = TAB_W - 44;
    cairo_select_font_face(cr, "DejaVu Serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    for (;;) {
        cairo_set_font_size(cr, s1);
        cairo_text_extents(cr, big, &e1);
        cairo_set_font_size(cr, s2);
        cairo_text_extents(cr, unit, &e2);
        if (e1.x_advance + e2.x_advance <= maxw || s1 < 24)
            break;
        s1 -= 1;
    }
    double x0 = TAB_W / 2.0 - (e1.x_advance + e2.x_advance) / 2, base = 50;
    rgb gold = { 1.0, 0.84, 0.52 }, dim = { 0.80, 0.62, 0.40 };
    for (int pass = 0; pass < 2; pass++) {
        cairo_set_font_size(cr, s1);
        cairo_move_to(cr, x0, base);
        cairo_text_path(cr, big);
        cairo_set_font_size(cr, s2);
        cairo_move_to(cr, x0 + e1.x_advance, base);
        cairo_text_path(cr, unit);
        if (pass == 0) {                                    /* chiselled shadow, then a warm glow */
            cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
            cairo_set_line_width(cr, 4);
            cairo_set_source_rgba(cr, 0.02, 0.01, 0, 0.9);
            cairo_stroke(cr);
        } else {
            cairo_set_source_rgb(cr, gold.r, gold.g, gold.b);
            cairo_fill(cr);
        }
    }

    /* the rotating line: label dim, value bright */
    cairo_select_font_face(cr, "DejaVu Serif Condensed", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    double fs = 22;
    cairo_text_extents_t l1, l2;
    for (;;) {
        cairo_set_font_size(cr, fs);
        cairo_text_extents(cr, lab, &l1);
        cairo_text_extents(cr, val, &l2);
        if (l1.x_advance + l2.x_advance <= maxw || fs <= 18)
            break;
        fs -= 0.5;
    }
    double lx = TAB_W / 2.0 - (l1.x_advance + l2.x_advance) / 2, ly = 76;
    for (int pass = 0; pass < 2; pass++) {
        cairo_move_to(cr, lx, ly);
        cairo_text_path(cr, lab);
        if (pass == 0) {
            cairo_move_to(cr, lx + l1.x_advance, ly);
            cairo_text_path(cr, val);
            cairo_set_line_width(cr, 3.5);
            cairo_set_source_rgba(cr, 0.02, 0.01, 0, 0.9);
            cairo_stroke(cr);
        } else {
            set_rgb(cr, dim);
            cairo_fill(cr);
            cairo_move_to(cr, lx + l1.x_advance, ly);
            cairo_text_path(cr, val);
            set_rgb(cr, gold);
            cairo_fill(cr);
        }
    }
    cairo_destroy(cr);
}

static void update_caption(void)
{
    static int last = -1;
    static const char *moods[] = { "quiet devotion", "evening chant", "high mass", "RAPTURE" };
    int m = W.hot ? 5 : W.idle ? 4 : W.mood;
    if (m == last)
        return;
    last = m;
    const char *txt = m == 5 ? "it is too hot" : m == 4 ? "the faithful sleep" : moods[m];
    cairo_t *cr = cairo_create(cap_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    rgb c = m == 5 ? (rgb){ 1.0, 0.45, 0.35 } : m == 3 ? (rgb){ 1.0, 0.82, 0.45 } : (rgb){ 0.86, 0.70, 0.48 };
    cairo_select_font_face(cr, "DejaVu Serif", CAIRO_FONT_SLANT_ITALIC, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 22);
    cairo_text_extents_t e;
    cairo_text_extents(cr, txt, &e);
    cairo_move_to(cr, CAP_W / 2.0 - e.width / 2 - e.x_bearing, CAP_H / 2.0 - e.height / 2 - e.y_bearing);
    cairo_text_path(cr, txt);
    cairo_set_line_width(cr, 4);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_source_rgba(cr, 0.02, 0.01, 0, 0.85);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
    cairo_destroy(cr);
}

static void render(cairo_t *cr, const stats *g, const sys_stats *s, double shown_tok, double t)
{
    static item items[80];
    int n = 0;

    cairo_set_source_surface(cr, bg_img, 0, 0);
    cairo_paint(cr);

    /* light: the monolith's halo (reddening with heat), the braziers' fire on the floor */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    double hi = 0.35 + 0.4 * W.act + 0.25 * W.pulse;
    double hh = W.heat * W.heat;
    blit(cr, halo_cool, MONO_X - sw(halo_cool) / 2.0, 205 - sh_(halo_cool) / 2.0, hi * (1 - hh));
    if (hh > 0.01)
        blit(cr, halo_hot, MONO_X - sw(halo_hot) / 2.0, 205 - sh_(halo_hot) / 2.0, (0.5 + 0.5 * hi) * hh * (0.85 + 0.15 * sin(t * 5)));
    for (int b = 0; b < 2; b++)
        blit(cr, fire_glow, BRZ_X[b] - 95, BRZ_Y - 110 - 95 + 30 * (1 - W.pw[b]), 0.15 + 0.75 * W.pw[b]);
    for (int i = 0; i < N_RUNES; i++)                       /* forks: runes flare on the rim */
        if (rune_lit[i] > 0.02)
            blit(cr, rune_spr[rune_g[i]], rune_x[i] - sw(rune_spr[0]) / 2.0, rune_y[i] - sh_(rune_spr[0]) / 2.0, rune_lit[i]);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* everything standing on the floor, back to front */
    for (int i = 0; i < N_CANDLES; i++)
        items[n++] = (item){ cand_y[i], K_CANDLE, i };
    for (int i = 0; i < MAX_ACO; i++)
        if (aco[i].state != A_OFF)
            items[n++] = (item){ aco[i].y, K_ACO, i };
    if (fanner.state != A_OFF)
        items[n++] = (item){ fanner.y, K_FAN, 0 };
    for (int j = 0; j < MAX_ZOMB; j++)
        if (zomb[j].state != A_OFF)
            items[n++] = (item){ zomb[j].y, K_ZOMB, j };
    items[n++] = (item){ BRZ_Y, K_BRAZ, 0 };
    items[n++] = (item){ BRZ_Y, K_BRAZ, 1 };
    items[n++] = (item){ GONG_Y, K_GONG, 0 };
    items[n++] = (item){ 190, K_BELL, 0 };
    items[n++] = (item){ MONO_BASE - 20, K_MONO, 0 };
    items[n++] = (item){ CENSER_Y, K_CENSER, 0 };
    items[n++] = (item){ CENSER_Y, K_CENSER, 1 };
    items[n++] = (item){ BOWL_Y, K_BOWL, 0 };
    qsort(items, n, sizeof(item), by_y);
    for (int i = 0; i < n; i++) {
        switch (items[i].kind) {
        case K_CANDLE: draw_candle(cr, items[i].idx, t); break;
        case K_ACO:    draw_acolyte(cr, &aco[items[i].idx]); break;
        case K_FAN:    draw_acolyte(cr, &fanner); break;
        case K_ZOMB:   draw_zombie(cr, &zomb[items[i].idx]); break;
        case K_BRAZ:   draw_brazier(cr, items[i].idx, t); break;
        case K_GONG:   draw_gong(cr, t); break;
        case K_BELL:   draw_bell(cr, t); break;
        case K_MONO:   draw_monolith(cr, t); break;
        case K_BOWL:   draw_bowl(cr, t); break;
        case K_CENSER: {
            double w = sw(censer_img), h = sh_(censer_img);
            blit(cr, censer_img, CENSER_X[items[i].idx] - w / 2, CENSER_Y - h, 1);
            break;
        }
        }
    }

    /* sleepers' zzz */
    for (int i = 0; i < MAX_ACO; i++) {
        const acolyte *a = &aco[i];
        if (a->state != A_SLEEP || a->frame != a->prev || a->fade < 1)
            continue;
        for (int k = 0; k < 2; k++) {
            double u = fmod(t * 0.35 + i * 0.37 + k * 0.5, 1.0);
            double kk = depth_k(a->y);
            double x = a->x + a->face * 10 * kk / 0.4 + u * 14 * a->face + sin(u * 6) * 3;
            double y = a->y - 70 * kk - u * 28;
            blit(cr, zzz_spr, x - 8, y - 9, sin(u * M_PI) * 0.9);
        }
    }

    /* incense */
    for (int i = 0; i < MAX_PUFF; i++) {
        const puff *p = &puffs[i];
        if (!p->alive)
            continue;
        double u = p->age / p->life;
        int si = (int)(u * (N_PUFF_SPR - 1) * p->size);
        si = si >= N_PUFF_SPR ? N_PUFF_SPR - 1 : si;
        cairo_surface_t *ps = puff_spr[si];
        blit(cr, ps, p->x - sw(ps) / 2.0, p->y - sh_(ps) / 2.0, 0.32 * sin(M_PI * fmin(1, u * 1.2)) * (0.5 + 0.5 * W.entropy));
    }

    /* offerings and sparks, additive */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = 0; i < MAX_GLYPH; i++) {
        const glyph *gl = &glyphs[i];
        if (!gl->alive)
            continue;
        double u = gl->t, v = 1 - u;
        double x = v * v * gl->x0 + 2 * v * u * gl->cx + u * u * gl->x1;
        double y = v * v * gl->y0 + 2 * v * u * gl->cy + u * u * gl->y1;
        double a = clamp01(u / 0.12) * clamp01((1 - u) / 0.25);
        cairo_surface_t *gs = glyph_spr[gl->g];
        double k = 0.7 + 0.3 * sin(u * M_PI);
        if (k > 0.97)
            blit(cr, gs, x - sw(gs) / 2.0, y - sh_(gs) / 2.0, a);
        else
            blit_scaled(cr, gs, x, y, k, k, a);
    }
    for (int i = 0; i < MAX_SPARK; i++) {
        const spark *p = &sparks[i];
        if (p->alive)
            blit(cr, spark_spr, p->x - 3, p->y - 3, 1 - p->age / p->life);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* carrier pigeons */
    for (int i = 0; i < MAX_PIGEON; i++) {
        const pigeon *p = &pigeons[i];
        if (!p->alive)
            continue;
        double u = p->t, x, y, k, a;
        if (!p->out) {                                      /* in from the left, into the monolith */
            x = -30 + (MONO_X - 6 + 30) * u;
            y = p->y0 + (112 - p->y0) * u * u - 18 * sin(u * M_PI);
            k = 0.24 - 0.1 * u * u;
            a = clamp01((1 - u) / 0.2);
        } else {                                            /* out of the monolith to the right */
            x = MONO_X + 6 + (510 - MONO_X) * u;
            y = 112 + (p->y1 - 112) * sqrt(u) - 14 * sin(u * M_PI);
            k = 0.14 + 0.1 * sqrt(u);
            a = clamp01(u / 0.2);
        }
        int f = (int)(p->t * p->dur * 9 + i) % 4;
        blit_scaled(cr, pigeon_src[f == 3 ? 1 : f], x, y, k, k, a);
    }

    /* text */
    update_caption();
    blit(cr, cap_cache, 240 - CAP_W / 2, CAP_Y, 1);
    update_hud(g, s, shown_tok, t);
    blit(cr, hud_cache, TAB_X, TAB_Y, 1);
}

/* ---------------------------------------------------------------- main */

static void world_reset(void)
{
    memset(&W, 0, sizeof(W));
    W.count = ACO_MIN;
    for (int i = 0; i < MAX_ACO; i++) {
        double off = aco[i].off;
        memset(&aco[i], 0, sizeof(acolyte));
        aco[i].off = off;
        aco[i].slot = i;
        /* the first ACO_MIN are already here when the display starts */
        if (i < ACO_MIN) {
            aco[i].state = A_KNEEL;
            aco[i].x = slot_x[i];
            aco[i].y = slot_y[i];
            aco[i].face = slot_x[i] < 240 ? 1 : -1;
            aco[i].frame = aco[i].prev = P_UP;
            aco[i].fade = 1;
        }
    }
    memset(&fanner, 0, sizeof(fanner));
    memset(zomb, 0, sizeof(zomb));
    memset(glyphs, 0, sizeof(glyphs));
    memset(puffs, 0, sizeof(puffs));
    memset(sparks, 0, sizeof(sparks));
    memset(pigeons, 0, sizeof(pigeons));
    memset(gong_rings, 0, sizeof(gong_rings));
    memset(bell_rings, 0, sizeof(bell_rings));
    memset(cand_l, 0, sizeof(cand_l));
}

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats g = { 0 };
    sys_stats sys = { 0 };
    double last, next_poll = 0, t0, shown_tok = 0;
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
    build_caches();
    world_reset();
    if (showcase) {                     /* the script opens at night: one sweeping, the rest asleep */
        W.idle = 1;
        for (int i = 0; i < ACO_MIN; i++) {
            aco[i].state = i == 0 ? A_TO_SWEEP : A_SLEEP;
            aco[i].frame = aco[i].prev = i == 0 ? P_UP : i & 1 ? P_CURL : P_SIT;
        }
    }

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* scenes from the showcase script, each held for 14 s so the cult settles */
        struct { double at; const char *png; } scenes[] = {
            { 27, "shrine_preview.png" },
            { 13, "shrine_mid.png" },
            { 3, "shrine_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(5 + k);
            world_reset();
            int n = 14 * FPS_BUSY;
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                double t = i / (double)FPS_BUSY;
                if (i == n - 60)
                    b0 = now_s();
                if (i % 12 == 0)
                    showcase_poll(&g, &sys, scenes[k].at + fmod(t, 1.0) * 0.01);
                shown_tok += (g.tok_s - shown_tok) * 0.15;
                simulate(&g, &sys, shown_tok, 1.0 / FPS_BUSY, t);
                render(cr, &g, &sys, shown_tok, t);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame (render + jpeg), jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / 60, len);
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
    while (!stop) {
        double t = now_s(), dt = t - last, k;
        last = t;
        if (dt > 0.25)
            dt = 0.25;

        if (t >= next_poll) {
            next_poll = t + (demo ? 0.5 : 1.0);
            if (showcase) {
                showcase_poll(&g, &sys, t - t0);
            } else if (demo) {
                demo_poll(&g, &sys, t - t0);
            } else {
                gpus_poll(&g);
                if (gpu_source)
                    gpu_rate_poll(&g);
                else
                    vllm_poll(&g, t);
                sys_poll(&sys, t);
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        shown_tok += (g.tok_s - shown_tok) * k;

        simulate(&g, &sys, shown_tok, dt, t - t0);
        render(cr, &g, &sys, shown_tok, t - t0);
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

        /* the night office runs slower */
        int idle = W.idle && !showcase;
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
