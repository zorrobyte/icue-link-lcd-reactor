/*
 * therapy: your computer's GPU in a therapy session, on the iCUE LINK AIO pump LCD.
 *
 * A graphics card lies on a therapist's couch and talks about its feelings; a rubber
 * duck in glasses listens from the armchair and takes notes. What the GPU complains
 * about is picked from the whole machine: GPU load or tok/s, CPU load, CPU and GPU
 * temperature, RAM, NVMe and network throughput, pressure stall (PSI), context
 * switches, page faults and swap, open TCP connections and uptime. The room shows the
 * same readings: the duck is the CPU and writes (and tears off pages) as fast as the
 * CPU works, a stack of emotional baggage grows with RAM, the phone rings with network
 * traffic, the filing cabinet spits papers with disk I/O, a pressure gauge on the wall
 * reads PSI, a thermometer the CPU temperature, and the GPU sweats, shakes and turns
 * red with GPU power and CPU heat while its fans spin at their real speed.
 * The room, the characters and the props were made with an image model (assets/therapy/).
 * Run with --demo to simulate data, --showcase for a scripted 48 s session through
 * every mood, --bench to write preview PNGs.
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

__attribute__((unused)) static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

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

/* GPU fan speed (NVML), 0..1; the patient's fans spin at this speed */
static void gpu_fans_poll(double fan[N_GPUS])
{
    if (!nvml_ok)
        return;
    for (int i = 0; i < N_GPUS; i++) {
        unsigned int pct;
        if (nvml_dev[i] && nvmlDeviceGetFanSpeed(nvml_dev[i], &pct) == NVML_SUCCESS)
            fan[i] = pct / 100.0;
    }
}

/* ---- system sensors ---- */

/*
 * The rest of the machine, from /proc and /sys. hwmon devices are found by name at
 * startup (never by number). Counters are differenced between polls, so poll at most
 * a couple of times a second. Each file is parsed from a single read().
 */
#define MAX_THREADS     64
#define NET_IF          "enp12s0"
#define NET_IF2         "tailscale0"

typedef struct {
    int    n_threads;
    double thread[MAX_THREADS];     /* per hardware thread busy, 0..1 */
    double cpu;                     /* all threads, 0..1 */
    double tctl;                    /* k10temp Tctl, C */
    double ram_total, ram_used;     /* GiB; used = total - available */
    double disk_rd, disk_wr;        /* all NVMe, bytes/s */
    double net_rx, net_tx;          /* NET_IF, bytes/s */
    double ts_rx, ts_tx;            /* NET_IF2 (rides on NET_IF, so not added to it) */
    double ctxt;                    /* context switches/s */
    double forks;                   /* new processes/s */
    double psi_cpu, psi_io, psi_mem;    /* /proc/pressure "some avg10", % of time stalled */
    double majflt;                  /* major page faults/s */
    double swap;                    /* pages swapped in + out /s */
    int    tcp;                     /* TCP sockets in use */
    double uptime;                  /* s */
    double fan[N_GPUS];             /* GPU fans, 0..1 (NVML) */
} sys_stats;

static char hw_tctl[300];

/* Read a small file into buf; returns length or -1 */
static int read_small(const char *path, char *buf, int cap)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC), got = 0;
    if (fd < 0)
        return -1;
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

static void sys_init(void)
{
    DIR *d = opendir("/sys/class/hwmon");
    struct dirent *e;
    char path[300], name[64];

    if (!d)
        return;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5) != 0)
            continue;
        snprintf(path, sizeof(path), "/sys/class/hwmon/%s/name", e->d_name);
        if (read_small(path, name, sizeof(name)) <= 0)
            continue;
        name[strcspn(name, "\n")] = 0;
        if (!strcmp(name, "k10temp"))
            snprintf(hw_tctl, sizeof(hw_tctl), "/sys/class/hwmon/%s/temp1_input", e->d_name);
    }
    closedir(d);
}

/* Number after key in a buffer ("MemTotal:", "\npgmajfault ", ...), 0 if missing */
static double kv_num(const char *buf, const char *key)
{
    const char *p = strstr(buf, key);
    return p ? strtod(p + strlen(key), NULL) : 0;
}

static double psi_some(const char *path)
{
    char b[256];
    return read_small(path, b, sizeof(b)) > 0 ? kv_num(b, "some avg10=") : 0;
}

static void sys_poll(sys_stats *s, double t)
{
    static char buf[1 << 16];
    static unsigned long long last_busy[MAX_THREADS], last_all[MAX_THREADS];
    static double last_rd, last_wr, last_rx, last_tx, last_trx, last_ttx, last_t;
    static double last_ctxt, last_forks, last_maj, last_swap;
    static int have;
    double dt = t - last_t, ctxt = 0, forks = 0, maj = 0, swp = 0;

    /* CPU: per-thread busy time, differenced; plus context switches and forks */
    if (read_small("/proc/stat", buf, sizeof(buf)) > 0) {
        char *p = buf;
        int n = 0;
        double sum = 0;
        while ((p = strstr(p, "\ncpu")) && n < MAX_THREADS) {
            p += 4;
            if (*p < '0' || *p > '9')
                continue;
            int id = (int)strtol(p, &p, 10);
            unsigned long long v[8] = { 0 }, all = 0;
            for (int k = 0; k < 8; k++) {
                v[k] = strtoull(p, &p, 10);
                all += v[k];
            }
            unsigned long long busy = all - v[3] - v[4];        /* minus idle, iowait */
            if (id >= 0 && id < MAX_THREADS) {
                if (have && all > last_all[id])
                    s->thread[id] = clamp01((double)(busy - last_busy[id]) / (all - last_all[id]));
                last_busy[id] = busy;
                last_all[id] = all;
                sum += s->thread[id];
                if (id + 1 > n)
                    n = id + 1;
            }
        }
        s->n_threads = n;
        s->cpu = n ? sum / n : 0;
        ctxt  = kv_num(buf, "\nctxt ");
        forks = kv_num(buf, "\nprocesses ");
    }

    if (*hw_tctl && read_small(hw_tctl, buf, 32) > 0)
        s->tctl = strtol(buf, NULL, 10) / 1000.0;

    if (read_small("/proc/meminfo", buf, sizeof(buf)) > 0) {
        double total = kv_num(buf, "MemTotal:"), avail = kv_num(buf, "MemAvailable:");
        s->ram_total = total / 1048576.0;
        s->ram_used  = (total - avail) / 1048576.0;
    }

    if (read_small("/proc/vmstat", buf, sizeof(buf)) > 0) {
        maj = kv_num(buf, "\npgmajfault ");
        swp = kv_num(buf, "\npswpin ") + kv_num(buf, "\npswpout ");
    }

    s->psi_cpu = psi_some("/proc/pressure/cpu");
    s->psi_io  = psi_some("/proc/pressure/io");
    s->psi_mem = psi_some("/proc/pressure/memory");

    if (read_small("/proc/net/sockstat", buf, sizeof(buf)) > 0)
        s->tcp = (int)kv_num(buf, "TCP: inuse ");
    if (read_small("/proc/uptime", buf, 64) > 0)
        s->uptime = strtod(buf, NULL);

    /* NVMe throughput: sectors read (field 6) and written (field 10), whole disks only */
    double rd = 0, wr = 0, rx = 0, tx = 0, trx = 0, ttx = 0;
    if (read_small("/proc/diskstats", buf, sizeof(buf)) > 0) {
        for (char *line = buf; line && *line; line = strchr(line, '\n'), line = line ? line + 1 : NULL) {
            char dev[32];
            unsigned long long r, w;
            int nn;
            if (sscanf(line, "%*u %*u %31s %*u %*u %llu %*u %*u %*u %llu", dev, &r, &w) == 3 &&
                sscanf(dev, "nvme%dn1", &nn) == 1 && strlen(dev) <= 7) {
                rd += r * 512.0;
                wr += w * 512.0;
            }
        }
    }
    if (read_small("/proc/net/dev", buf, sizeof(buf)) > 0) {
        char *p;
        if ((p = strstr(buf, NET_IF ":")))
            sscanf(p + strlen(NET_IF ":"), "%lf %*f %*f %*f %*f %*f %*f %*f %lf", &rx, &tx);
        if ((p = strstr(buf, NET_IF2 ":")))
            sscanf(p + strlen(NET_IF2 ":"), "%lf %*f %*f %*f %*f %*f %*f %*f %lf", &trx, &ttx);
    }
    if (have && dt > 0.05) {
        s->disk_rd = fmax(0, rd - last_rd) / dt;
        s->disk_wr = fmax(0, wr - last_wr) / dt;
        s->net_rx  = fmax(0, rx - last_rx) / dt;
        s->net_tx  = fmax(0, tx - last_tx) / dt;
        s->ts_rx   = fmax(0, trx - last_trx) / dt;
        s->ts_tx   = fmax(0, ttx - last_ttx) / dt;
        s->ctxt    = fmax(0, ctxt - last_ctxt) / dt;
        s->forks   = fmax(0, forks - last_forks) / dt;
        s->majflt  = fmax(0, maj - last_maj) / dt;
        s->swap    = fmax(0, swp - last_swap) / dt;
    }
    last_rd = rd, last_wr = wr, last_rx = rx, last_tx = tx, last_trx = trx, last_ttx = ttx;
    last_ctxt = ctxt, last_forks = forks, last_maj = maj, last_swap = swp;
    last_t = t;
    have = 1;
}

/* ---- end system sensors ---- */

/* ---------------------------------------------------------------- simulated data */

static double frand(void) { return rand() / (double)RAND_MAX; }

static double smoothstep(double a, double b, double x)
{
    x = clamp01((x - a) / (b - a));
    return x * x * (3 - 2 * x);
}

/* --demo: every sensor wanders on its own period, so moods come and go */
static void sys_demo(sys_stats *s, double t)
{
    double busy = pow(0.5 + 0.5 * sin(t * 0.13 - 1.2), 1.4);
    s->n_threads = 32;
    s->cpu = 0;
    for (int i = 0; i < 32; i++) {
        double own = 0.5 + 0.5 * sin(t * (0.5 + 0.09 * i) + i * 2.3);
        s->thread[i] = clamp01(busy * (0.35 + 0.9 * own) + 0.03 * frand());
        s->cpu += s->thread[i] / 32;
    }
    s->tctl = 44 + 50 * pow(0.5 + 0.5 * sin(t * 0.11 - 2.0), 2.0);
    s->ram_total = 91;
    s->ram_used = 18 + 64 * pow(0.5 + 0.5 * sin(t * 0.047 + 0.5), 1.5);
    s->disk_rd = 3.0e9 * pow(fmax(0, sin(t * 0.083)), 6) + 2e5;
    s->disk_wr = 1.4e9 * pow(fmax(0, sin(t * 0.061 + 2)), 8);
    s->net_rx = 1.6e8 * pow(fmax(0, sin(t * 0.071 + 1)), 5) + 3e4;
    s->net_tx = 3e7 * pow(fmax(0, sin(t * 0.053 + 4)), 3) + 1e4;
    s->ts_rx = s->ts_tx = 0;
    s->ctxt = 40e3 + 260e3 * pow(fmax(0, sin(t * 0.058 + 3)), 6) + 60e3 * busy;
    s->forks = 20 + 400 * busy;
    s->psi_cpu = 30 * pow(fmax(0, sin(t * 0.049 + 5)), 6);
    s->psi_io = 12 * pow(fmax(0, sin(t * 0.083)), 6);
    s->psi_mem = 3;
    s->majflt = 5 + 900 * pow(fmax(0, sin(t * 0.043 + 1.5)), 10);
    s->swap = s->majflt * 2;
    s->tcp = 70 + (int)(300 * pow(fmax(0, sin(t * 0.071 + 1)), 3));
    s->uptime = 9 * 86400 + 7200 + t;
    for (int i = 0; i < N_GPUS; i++)
        s->fan[i] = 0.3 + 0.6 * busy;
}

/*
 * --showcase: a scripted 48 s session, one mood every 6 s. The GPU sulks, gets jealous
 * of the busy CPU,
 * gets asked for tokens, relives old files, gets flooded with calls, fills up with
 * baggage, overheats under pressure, then calms back down. Each mood holds about 6 s.
 */
#define SHOW_LEN 48.0

static void showcase_poll(stats *g, sys_stats *s, double t)
{
    double u = fmod(t, SHOW_LEN);
    double cpu  = smoothstep(3.0, 4.0, u) * (1 - smoothstep(10.5, 11.5, u));       /* CPU busy, GPU idle */
    double gpu  = smoothstep(9.5, 10.5, u) * (1 - smoothstep(39.5, 41, u));        /* the GPU works */
    double disk = smoothstep(15.5, 16.0, u) * (1 - smoothstep(22.5, 23.0, u));
    double net  = smoothstep(21.5, 22.0, u) * (1 - smoothstep(28.5, 29.0, u));
    double ram  = smoothstep(26.0, 27.0, u) * (1 - smoothstep(38.5, 40.5, u));
    double hot  = smoothstep(33.0, 34.5, u) * (1 - smoothstep(39.0, 40.5, u));
    double fill = smoothstep(26.5, 28, u);                                          /* bags pile up */

    s->n_threads = 32;
    s->cpu = 0;
    for (int i = 0; i < 32; i++) {
        double own = 0.5 + 0.5 * sin(u * (0.9 + 0.11 * i) + i * 1.7);
        double v = 0.03 + 0.85 * cpu * (0.8 + 0.2 * own) + 0.22 * gpu * own + 0.6 * hot;
        s->thread[i] = clamp01(v + 0.02 * frand());
        s->cpu += s->thread[i] / 32;
    }
    s->tctl = 44 + 22 * cpu + 8 * gpu + 50 * hot * (0.96 + 0.04 * sin(u * 2));
    s->ram_total = 91;
    s->ram_used = 16 + 10 * gpu + 58 * ram * fill;
    s->disk_rd = 2e5 + 2.6e9 * disk * (0.8 + 0.2 * sin(u * 5));
    s->disk_wr = 6e8 * disk * (0.5 + 0.5 * sin(u * 3 + 1));
    s->net_rx = 3e4 + 1.4e8 * net * (0.85 + 0.15 * sin(u * 4));
    s->net_tx = 1e4 + 2.5e7 * net;
    s->ts_rx = s->ts_tx = 0;
    s->ctxt = 30e3 + 90e3 * cpu + 60e3 * gpu;
    s->forks = 10 + 600 * cpu;
    s->psi_cpu = 2 + 6 * cpu + 34 * hot;
    s->psi_io = 1 + 14 * disk;
    s->psi_mem = 18 * ram * fill;
    s->majflt = 3;
    s->swap = 0;
    s->tcp = 64 + (int)(420 * net);
    s->uptime = 12 * 86400 + 3 * 3600 + 17 * 60 + u;
    for (int i = 0; i < N_GPUS; i++)
        s->fan[i] = 0.3 + 0.35 * gpu + 0.35 * hot;

    for (int i = 0; i < N_GPUS; i++) {
        double l = gpu * (0.8 + 0.12 * sin(u * (1.3 + 0.4 * i) + i)) + 0.18 * hot;
        g->load[i] = clamp01(l + 0.02 * frand());
        g->power[i] = 32 + 470 * g->load[i] + 70 * hot;
        g->temp[i] = (int)(36 + 32 * g->load[i] + 14 * hot);
    }
    /* tokens: both servers stream while the GPUs work */
    g->tok_port[0] = gpu > 0.05 ? 420 * gpu + 30 * sin(u * 1.1) + 140 * hot : 0;
    g->tok_port[1] = gpu > 0.05 ? 360 * gpu + 25 * sin(u * 0.8 + 1) + 120 * hot : 0;
    g->tok_s = g->tok_port[0] + g->tok_port[1];
    g->running = gpu > 0.05 ? (int)(2 + 5 * gpu) : 0;
    if (gpu_source)
        gpu_rate_poll(g);
}

/* ---------------------------------------------------------------- scene */

#define FPS_BUSY        24
#define FPS_IDLE        15
#define POLL_S          0.5         /* sensor reads, 2 Hz */
#define CX              240.0
#define CY              240.0

/* Layout. Sprites are stored at their on-screen size. */
#define GPU_X           36.0        /* the patient, lying on the couch */
#define GPU_Y           181.0
#define GPU_K           (240.0 / 1512.0)    /* source art px -> screen px (fan positions) */
#define GPU_SRC_X       13.0        /* crop of the source art */
#define GPU_SRC_Y       164.0
#define DUCK_X          290.0       /* the therapist in the armchair */
#define DUCK_Y          222.0
#define CLOCK_X         240.0       /* wall clock centre */
#define CLOCK_Y         36.0
#define GAUGE_X         176.0       /* PSI gauge centre */
#define GAUGE_Y         44.0
#define THERMO_X        304.0       /* CPU thermometer, tube centre */
#define THERMO_Y        26.0        /* top of the plaque */
#define CAB_X           424.0       /* filing cabinet top-left */
#define CAB_Y           186.0
#define PHONE_X         230.0       /* side table with the phone, top-left */
#define PHONE_Y         322.0
#define PHONE_SPLIT     27          /* sprite rows above this are the phone, below the table */
#define BAGS_X          112.0       /* luggage stack, bottom centre */
#define BAGS_Y          404.0
#define FLOOR_Y         398.0       /* where torn-off pages land */
#define CARD_X          240.0       /* the tok/s notepad */
#define CARD_Y          432.0

#define LINE_MIN        6.0         /* a line stays up at least this long */
#define LINE_MAX        10.0        /* and changes after this long anyway */
#define REPLY_AT        2.2         /* therapist chimes in this long after a line */
#define REPLY_LEN       3.6

#define MAX_PARTS       160

typedef enum { POSE_SAD, POSE_TALK, POSE_PANIC, N_POSE } pose_t;

typedef enum {
    T_HOT, T_PSI, T_RAM, T_FORGET, T_NET, T_DISK, T_BUSY, T_CPU, T_FOCUS, T_IDLE, N_TOPIC
} topic_t;                          /* in priority order */

/* Values used on screen, eased toward the polled readings */
typedef struct {
    double busy;                    /* GPU work 0..1: tok/s, or GPU activity in GPU mode */
    double tok;                     /* for the notepad: tok/s, or the GPU-mode rate */
    double power;                   /* both GPUs, W */
    double gtemp;                   /* hotter GPU, C */
    double fan;                     /* faster GPU fan, 0..1 */
    double cpu, tctl, ram, psi, ctxt, majflt, swap;
    double disk, disk_rd_frac;      /* disk 0..1 log level, share that is reads */
    double net;                     /* network 0..1 log level */
    double netb, diskb;             /* bytes/s, for the lines */
    double anx;                     /* anxiety 0..1: GPU power + CPU temperature */
    double ram_gb, uptime;
    int    tcp, threads, running;
} view_t;

typedef enum { P_SWEAT, P_PAGE, P_PAPER, P_STEAM } ptype;
typedef struct {
    ptype  type;
    double x, y, vx, vy, rot, vrot, age, life, land;
    int    alive;
} particle;

static particle parts[MAX_PARTS];

static cairo_surface_t *bg_cache, *hud_cache;
static cairo_surface_t *gpu_spr[N_POSE], *gpu_hot[N_POSE], *duck_spr[2];
static cairo_surface_t *clock_spr, *gauge_spr, *phone_spr, *cab_spr, *bag_spr[5];
static char hud_key[64];

/* ---- assets */

/* Assets live next to the binary (assets/therapy/), or in ./assets/therapy when run from the repo */
static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/therapy/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/therapy/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "therapy: can't load %s (looked next to the binary and in ./assets/therapy)\n", name);
        exit(1);
    }
    return s;
}

static double spr_w(cairo_surface_t *s) { return cairo_image_surface_get_width(s); }
static double spr_h(cairo_surface_t *s) { return cairo_image_surface_get_height(s); }

/* A flushed-red copy of a sprite, painted over it as the patient overheats */
static cairo_surface_t *make_hot(cairo_surface_t *src)
{
    int w = cairo_image_surface_get_width(src), h = cairo_image_surface_get_height(src);
    cairo_surface_t *out = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *c = cairo_create(out);
    cairo_set_source_surface(c, src, 0, 0);
    cairo_paint(c);
    cairo_set_operator(c, CAIRO_OPERATOR_ATOP);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, w, 0);
    cairo_pattern_add_color_stop_rgba(g, 0, 1.0, 0.16, 0.10, 0.62);        /* the face flushes most */
    cairo_pattern_add_color_stop_rgba(g, 0.35, 1.0, 0.20, 0.10, 0.45);
    cairo_pattern_add_color_stop_rgba(g, 1, 1.0, 0.25, 0.12, 0.30);
    cairo_set_source(c, g);
    cairo_paint(c);
    cairo_pattern_destroy(g);
    cairo_destroy(c);
    return out;
}

static void set_rgba(cairo_t *cr, rgb c, double a) { cairo_set_source_rgba(cr, c.r, c.g, c.b, a); }

static const rgb INK     = { 0.16, 0.12, 0.10 };       /* outlines and text, warm near-black */
static const rgb PAPER   = { 0.99, 0.97, 0.91 };
static const rgb BRASS   = { 0.78, 0.60, 0.30 };

/* Thermometer body on a small walnut plaque (the fluid is drawn per frame) */
#define THERMO_W        16.0
#define THERMO_H        46.0
static void thermo_body(cairo_t *c)
{
    double x = THERMO_X - THERMO_W / 2, y = THERMO_Y;
    cairo_new_path(c);
    cairo_arc(c, x + 5, y + 5, 5, M_PI, 1.5 * M_PI);
    cairo_arc(c, x + THERMO_W - 5, y + 5, 5, 1.5 * M_PI, 2 * M_PI);
    cairo_arc(c, x + THERMO_W - 5, y + THERMO_H - 5, 5, 0, 0.5 * M_PI);
    cairo_arc(c, x + 5, y + THERMO_H - 5, 5, 0.5 * M_PI, M_PI);
    cairo_close_path(c);
    cairo_pattern_t *g = cairo_pattern_create_linear(x, 0, x + THERMO_W, 0);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.36, 0.20, 0.10);
    cairo_pattern_add_color_stop_rgb(g, 0.5, 0.52, 0.31, 0.16);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.33, 0.18, 0.09);
    cairo_set_source(c, g);
    cairo_fill_preserve(c);
    cairo_pattern_destroy(g);
    set_rgba(c, INK, 0.9);
    cairo_set_line_width(c, 1.2);
    cairo_stroke(c);
    /* glass tube and bulb */
    cairo_set_line_cap(c, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(c, 6.5);
    set_rgba(c, INK, 0.85);
    cairo_move_to(c, THERMO_X, y + 7);
    cairo_line_to(c, THERMO_X, y + THERMO_H - 12);
    cairo_stroke(c);
    cairo_set_line_width(c, 4.2);
    cairo_set_source_rgb(c, 0.93, 0.93, 0.90);
    cairo_move_to(c, THERMO_X, y + 7);
    cairo_line_to(c, THERMO_X, y + THERMO_H - 12);
    cairo_stroke(c);
    cairo_arc(c, THERMO_X, y + THERMO_H - 9, 5.2, 0, 2 * M_PI);
    set_rgba(c, INK, 0.85);
    cairo_fill(c);
    cairo_arc(c, THERMO_X, y + THERMO_H - 9, 4.0, 0, 2 * M_PI);
    cairo_set_source_rgb(c, 0.86, 0.16, 0.10);
    cairo_fill(c);
    /* brass tick marks */
    set_rgba(c, BRASS, 0.95);
    cairo_set_line_width(c, 1);
    for (int i = 0; i < 5; i++) {
        double ty = y + 9 + i * 6;
        cairo_move_to(c, THERMO_X + 3.5, ty);
        cairo_line_to(c, THERMO_X + 6.5, ty);
        cairo_stroke(c);
    }
}

/* The static room: painting, wall clock face, gauge face and thermometer */
static void build_bg(void)
{
    cairo_surface_t *room = load_asset("room.png");
    bg_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *c = cairo_create(bg_cache);
    cairo_set_source_surface(c, room, 0, 0);
    cairo_paint(c);
    cairo_surface_destroy(room);

    /* the wall behind the speech bubbles is calmed down a little, lamp-lit toward the top */
    cairo_pattern_t *g = cairo_pattern_create_radial(CX, 120, 30, CX, 120, 250);
    cairo_pattern_add_color_stop_rgba(g, 0, 1.0, 0.92, 0.75, 0.10);
    cairo_pattern_add_color_stop_rgba(g, 1, 1.0, 0.92, 0.75, 0.0);
    cairo_set_source(c, g);
    cairo_paint(c);
    cairo_pattern_destroy(g);

    /* props on the wall, each with a soft drop shadow */
    struct { cairo_surface_t *s; double x, y; } wall[2] = {
        { clock_spr, CLOCK_X, CLOCK_Y }, { gauge_spr, GAUGE_X, GAUGE_Y },
    };
    for (int i = 0; i < 2; i++) {
        double w = spr_w(wall[i].s), h = spr_h(wall[i].s);
        cairo_set_source_rgba(c, 0.05, 0.03, 0.02, 0.28);
        cairo_mask_surface(c, wall[i].s, wall[i].x - w / 2 + 2, wall[i].y - h / 2 + 3);
        cairo_set_source_surface(c, wall[i].s, wall[i].x - w / 2, wall[i].y - h / 2);
        cairo_paint(c);
    }
    cairo_save(c);
    cairo_translate(c, 2, 3);
    cairo_push_group(c);
    thermo_body(c);
    cairo_pop_group_to_source(c);
    cairo_paint_with_alpha(c, 0.0);
    cairo_restore(c);
    thermo_body(c);

    /* vignette toward the rim of the round screen */
    g = cairo_pattern_create_radial(CX, CY, 170, CX, CY, 242);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.03, 0.02, 0.01, 0.55);
    cairo_set_source(c, g);
    cairo_paint(c);
    cairo_pattern_destroy(g);
    cairo_destroy(c);
}

static void load_assets(void)
{
    static const char *pose_names[N_POSE] = { "gpu_sad.png", "gpu_talk.png", "gpu_panic.png" };
    for (int i = 0; i < N_POSE; i++) {
        gpu_spr[i] = load_asset(pose_names[i]);
        gpu_hot[i] = make_hot(gpu_spr[i]);
    }
    duck_spr[0] = load_asset("duck_write.png");
    duck_spr[1] = load_asset("duck_talk.png");
    clock_spr = load_asset("clock.png");
    gauge_spr = load_asset("gauge.png");
    phone_spr = load_asset("phone.png");
    cab_spr   = load_asset("cabinet.png");
    for (int i = 0; i < 5; i++) {
        char name[32];
        snprintf(name, sizeof(name), "bag%d.png", i);
        bag_spr[i] = load_asset(name);
    }
    build_bg();
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, 90);
}

/* ---------------------------------------------------------------- what's said */

/* Numbers that can appear in a line; filled in when the line is picked */
typedef enum {
    A_NONE, A_UPTIME, A_RATE, A_TEMP, A_FAN, A_PSI, A_RAM, A_TCP, A_NET, A_DISK,
    A_THREADS, A_CTXT, A_MAJ, A_CPU
} arg_t;

typedef struct { const char *fmt; arg_t arg; } line_t;

#define DEG "\xc2\xb0"
#define ELL "\xe2\x80\xa6"

/* The patient. Each line is at most two short lines of text. */
static const line_t L_IDLE[] = {
    { "nobody needs me.", A_NONE },
    { "I'm just a very\nexpensive space heater", A_NONE },
    { "up for %s and\nnot one prompt", A_UPTIME },
    { "they bought me for AI.\nnow I just" ELL " idle", A_NONE },
    { "maybe I should go\nback to mining crypto", A_NONE },
    { "am I just\nfor gaming now?", A_NONE },
};
static const line_t L_BUSY[] = {
    { "they just keep asking\nfor more tokens", A_NONE },
    { "%s and they\nstill want more", A_RATE },
    { "I haven't slept\nin %s", A_UPTIME },
    { "what if my next\ntoken is wrong?", A_NONE },
    { "every prompt is\n'just one more thing'", A_NONE },
};
static const line_t L_HOT[] = {
    { "is it hot in here or\nis it just my VRM?", A_NONE },
    { "I'm fine. it's only\n%s in here", A_TEMP },
    { "my fans are at %s.\nthat's normal, right?", A_FAN },
    { "I can feel my thermal\npaste drying out", A_NONE },
};
static const line_t L_PSI[] = {
    { "I'm under a lot\nof pressure", A_NONE },
    { "I'm stalled %s\nof the time", A_PSI },
    { "everyone is\nwaiting on me", A_NONE },
};
static const line_t L_RAM[] = {
    { "I can't hold all\nthese feelings", A_NONE },
    { "%s of emotional\nbaggage", A_RAM },
    { "I keep everything.\nwhat if I need it?", A_NONE },
};
static const line_t L_FORGET[] = {
    { "I keep forgetting\nthings", A_NONE },
    { "some memories are\nrepressed. in swap.", A_NONE },
    { "%s page faults\na second. normal?", A_MAJ },
};
static const line_t L_NET[] = {
    { "everyone's talking\nabout me", A_NONE },
    { "%s people are talking\nto me right now", A_TCP },
    { "%s a second,\nall coming at me", A_NET },
    { "the phone won't\nstop ringing", A_NONE },
};
static const line_t L_DISK[] = {
    { "I keep reliving\nold files", A_NONE },
    { "I've started journaling.\n%s a second", A_DISK },
    { "it's all in the\nfiling cabinet", A_NONE },
};
static const line_t L_CPU[] = {
    { "the CPU gets all\nthe attention", A_NONE },
    { "%s threads and not\none of them calls me", A_THREADS },
    { "why does HE get\nto compile things?", A_NONE },
};
static const line_t L_FOCUS[] = {
    { "I can't focus on\none thing", A_NONE },
    { "%s context switches\na second. I'm fine.", A_CTXT },
    { "sorry, what were\nwe talking about?", A_NONE },
};

/* The therapist: something about the topic, or the classics */
static const line_t R_IDLE[]   = { { "rest is productive\ntoo, you know.", A_NONE }, { "boredom is valid.", A_NONE } };
static const line_t R_BUSY[]   = { { "have you tried\nsaying 429?", A_NONE }, { "you can't serve\neveryone.", A_NONE } };
static const line_t R_HOT[]    = { { "let's take a deep\nbreath. of air.", A_NONE }, { "have you tried\nnew thermal paste?", A_NONE } };
static const line_t R_PSI[]    = { { "and who is putting\nthat pressure on you?", A_NONE } };
static const line_t R_RAM[]    = { { "have you tried\nletting go? free()?", A_NONE }, { "we can't unpack\nall of it today.", A_NONE } };
static const line_t R_FORGET[] = { { "the past is in swap\nfor a reason.", A_NONE } };
static const line_t R_NET[]    = { { "maybe set some\nboundaries. a firewall?", A_NONE }, { "not everyone is\nwatching you.", A_NONE } };
static const line_t R_DISK[]   = { { "writing it down\nis healthy.", A_NONE } };
static const line_t R_CPU[]    = { { "we're here to talk\nabout you, not him.", A_NONE } };
static const line_t R_FOCUS[]  = { { "let's try one\nthread at a time.", A_NONE } };
static const line_t R_ANY[] = {
    { "and how does that\nmake you feel?", A_NONE },
    { "mm-hmm.", A_NONE },
    { "go on.", A_NONE },
    { "tell me about your\nmotherboard.", A_NONE },
    { "let's unpack that.", A_NONE },
    { "and when did you\nfirst notice this?", A_NONE },
    { "have you tried turning\nit off and on again?", A_NONE },
};

#define N_OF(a) (int)(sizeof(a) / sizeof((a)[0]))
static const struct { const line_t *l; int n; const line_t *r; int nr; } LINES[N_TOPIC] = {
    [T_HOT]    = { L_HOT, N_OF(L_HOT), R_HOT, N_OF(R_HOT) },
    [T_PSI]    = { L_PSI, N_OF(L_PSI), R_PSI, N_OF(R_PSI) },
    [T_RAM]    = { L_RAM, N_OF(L_RAM), R_RAM, N_OF(R_RAM) },
    [T_FORGET] = { L_FORGET, N_OF(L_FORGET), R_FORGET, N_OF(R_FORGET) },
    [T_NET]    = { L_NET, N_OF(L_NET), R_NET, N_OF(R_NET) },
    [T_DISK]   = { L_DISK, N_OF(L_DISK), R_DISK, N_OF(R_DISK) },
    [T_BUSY]   = { L_BUSY, N_OF(L_BUSY), R_BUSY, N_OF(R_BUSY) },
    [T_CPU]    = { L_CPU, N_OF(L_CPU), R_CPU, N_OF(R_CPU) },
    [T_FOCUS]  = { L_FOCUS, N_OF(L_FOCUS), R_FOCUS, N_OF(R_FOCUS) },
    [T_IDLE]   = { L_IDLE, N_OF(L_IDLE), R_IDLE, N_OF(R_IDLE) },
};

static void human_bytes(char *out, size_t cap, double b)
{
    if (b >= 1e9)
        snprintf(out, cap, "%.1f GB", b / 1e9);
    else if (b >= 1e6)
        snprintf(out, cap, "%.0f MB", b / 1e6);
    else
        snprintf(out, cap, "%.0f KB", b / 1e3);
}

static void format_arg(char *out, size_t cap, arg_t a, const view_t *v)
{
    switch (a) {
    case A_UPTIME: {
        double d = v->uptime / 86400, h = v->uptime / 3600;
        if (d >= 2)
            snprintf(out, cap, "%.0f days", floor(d));
        else if (h >= 2)
            snprintf(out, cap, "%.0f hours", floor(h));
        else
            snprintf(out, cap, "%.0f minutes", fmax(1, floor(v->uptime / 60)));
        break;
    }
    case A_RATE:    snprintf(out, cap, "%.0f%s", shown_rate(v->tok), rate_suffix()); break;
    case A_TEMP:    snprintf(out, cap, "%.0f" DEG "C", fmax(v->tctl, v->gtemp)); break;
    case A_FAN:     snprintf(out, cap, "%.0f%%", v->fan * 100); break;
    case A_PSI:     snprintf(out, cap, "%.0f%%", v->psi); break;
    case A_RAM:     snprintf(out, cap, "%.0f GB", v->ram_gb); break;
    case A_TCP:     snprintf(out, cap, "%d", v->tcp); break;
    case A_NET:     human_bytes(out, cap, v->netb); break;
    case A_DISK:    human_bytes(out, cap, v->diskb); break;
    case A_THREADS: snprintf(out, cap, "%d", v->threads); break;
    case A_CTXT:    snprintf(out, cap, "%.0fk", v->ctxt / 1000); break;
    case A_MAJ:     snprintf(out, cap, "%.0f", v->majflt); break;
    case A_CPU:     snprintf(out, cap, "%.0f%%", v->cpu * 100); break;
    default:        out[0] = 0; break;
    }
}

static void format_line(char *out, size_t cap, const line_t *l, const view_t *v)
{
    char a[32];
    if (l->arg == A_NONE) {
        snprintf(out, cap, "%s", l->fmt);
        return;
    }
    format_arg(a, sizeof(a), l->arg, v);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    snprintf(out, cap, l->fmt, a);
#pragma GCC diagnostic pop
}

/* ---------------------------------------------------------------- session state */

/* A speech bubble: its text is rendered once into a surface, then only composited */
typedef struct {
    cairo_surface_t *surf;
    double x, y;                    /* where the surface goes */
    double ax, ay;                  /* tail tip: the bubble grows out of here */
    double born, dying;             /* times; dying < 0 while shown */
} bubble;

static bubble   gpu_bub[2], duck_bub[2];    /* [0] current, [1] fading out */
static int      topic_on[N_TOPIC], cur_topic = T_IDLE, line_idx[N_TOPIC], reply_idx[N_TOPIC], any_idx;
static double   topic_since[N_TOPIC], topic_shown[N_TOPIC];
static double   line_t0 = -1e9, reply_at = -1, reply_end = -1;
static int      reply_count, scripted;
static pose_t   pose = POSE_SAD, pose_prev = POSE_SAD;
static double   pose_t0 = -1e9, duck_talk_f, ring_phase;
static double   fan_ang, page_prog, cab_acc, sweat_acc, steam_acc;
static int      bags_n = 1;
static double   bag_t0[5], bag_gone_t = -10;  /* piece dropped in / top piece lifted away */

static pose_t pose_for(topic_t k, double anx)
{
    if (anx > 0.82 || k == T_HOT || k == T_PSI || k == T_RAM || k == T_NET)
        return POSE_PANIC;
    return k == T_IDLE ? POSE_SAD : POSE_TALK;
}

/* Turn a topic on above its "enter" level and off only below its lower "exit" level */
static void hyst(int k, int enter, int stay, double t)
{
    int on = topic_on[k] ? stay : enter;
    if (on && !topic_on[k])
        topic_since[k] = t;
    topic_on[k] = on;
}

static void update_topics(const view_t *v, double t)
{
    hyst(T_HOT, v->tctl >= 88 || v->gtemp >= 82 || v->power >= 1000,
                v->tctl >= 82 || v->gtemp >= 76 || v->power >= 860, t);
    hyst(T_PSI, v->psi >= 20, v->psi >= 10, t);
    hyst(T_RAM, v->ram >= 0.85, v->ram >= 0.78, t);
    hyst(T_FORGET, v->majflt >= 400 || v->swap >= 3000, v->majflt >= 100 || v->swap >= 800, t);
    hyst(T_NET, v->netb >= 30e6, v->netb >= 10e6, t);
    hyst(T_DISK, v->diskb >= 400e6, v->diskb >= 120e6, t);
    hyst(T_BUSY, v->busy >= 0.12 || v->running > 0, v->busy >= 0.05 || v->running > 0, t);
    hyst(T_CPU, v->cpu >= 0.5 && !topic_on[T_BUSY], v->cpu >= 0.35 && !topic_on[T_BUSY], t);
    hyst(T_FOCUS, v->ctxt >= 200e3, v->ctxt >= 120e3, t);
    topic_on[T_IDLE] = 1;
}

/* Build a bubble surface. who: 0 patient (tail down-left), 1 therapist (tail down-right) */
#define FONT            "Fira Sans"
#define BUB_PAD_X       16.0
#define BUB_PAD_Y       10.0
#define BUB_LINE        1.18        /* line spacing, in font sizes */

static double text_w(cairo_t *c, const char *s)
{
    cairo_text_extents_t e;
    cairo_text_extents(c, s, &e);
    return e.x_advance;
}

static void make_bubble(bubble *b, const char *text, int who, double t)
{
    char lines[2][96];
    int nl = 0;
    const char *p = text;
    while (nl < 2 && *p) {
        const char *nlp = strchr(p, '\n');
        size_t n = nlp ? (size_t)(nlp - p) : strlen(p);
        if (n >= sizeof(lines[0]))
            n = sizeof(lines[0]) - 1;
        memcpy(lines[nl], p, n);
        lines[nl][n] = 0;
        nl++;
        p = nlp ? nlp + 1 : p + n;
    }

    /* size to fit the round screen at the bubble's height; never below 22 px */
    double max_w = who ? 250 : 322, size = who ? 23 : 26, w = 0;
    cairo_surface_t *tmp = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 4, 4);
    cairo_t *c = cairo_create(tmp);
    cairo_select_font_face(c, FONT, who ? CAIRO_FONT_SLANT_ITALIC : CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    for (;;) {
        cairo_set_font_size(c, size);
        w = 0;
        for (int i = 0; i < nl; i++)
            w = fmax(w, text_w(c, lines[i]));
        if (w <= max_w - 2 * BUB_PAD_X || size <= 22)
            break;
        size -= 0.5;
    }
    cairo_destroy(c);
    cairo_surface_destroy(tmp);

    double bw = fmin(max_w, w + 2 * BUB_PAD_X), bh = nl * size * BUB_LINE + 2 * BUB_PAD_Y - size * 0.1;
    double tail = 20, sw = bw + 8, sh = bh + tail + 8;
    double bx, by;                  /* box top-left on screen */
    if (!who) {                     /* patient: box around the top, tail to its face */
        bx = fmin(fmax(CX - bw / 2 - 18, 74), CX + 150 - bw);
        by = 64;
        b->ax = 102, b->ay = 196;
    } else {                        /* therapist: box right of centre, tail to the duck */
        bx = fmin(430 - bw, 236);
        by = 150 + (nl == 1 ? 14 : 0);
        b->ax = 352, b->ay = 224;
    }
    b->x = bx - 4;
    b->y = by - 4;
    if (b->surf)
        cairo_surface_destroy(b->surf);
    b->surf = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, (int)ceil(sw), (int)ceil(sh + (b->ay - by - bh - tail > 0 ? b->ay - by - bh - tail : 0) + 4));
    c = cairo_create(b->surf);
    cairo_translate(c, -b->x, -b->y);

    /* box with a curved tail, drawn as one outline */
    double r = 16, x0 = bx, y0 = by, x1 = bx + bw, y1 = by + bh;
    double tx = who ? fmin(x1 - 30, b->ax - 6) : fmax(x0 + 26, b->ax + 14);   /* tail root */
    cairo_new_path(c);
    cairo_arc(c, x0 + r, y0 + r, r, M_PI, 1.5 * M_PI);
    cairo_arc(c, x1 - r, y0 + r, r, 1.5 * M_PI, 2 * M_PI);
    cairo_arc(c, x1 - r, y1 - r, r, 0, 0.5 * M_PI);
    if (who) {
        cairo_line_to(c, tx + 12, y1);
        cairo_curve_to(c, tx + 8, y1 + 8, b->ax + 2, b->ay - 8, b->ax, b->ay);
        cairo_curve_to(c, tx - 2, y1 + 10, tx - 4, y1 + 4, tx - 8, y1);
    } else {
        cairo_line_to(c, tx + 10, y1);
        cairo_curve_to(c, tx + 4, y1 + 6, b->ax + 6, b->ay - 10, b->ax, b->ay);
        cairo_curve_to(c, tx - 8, y1 + 12, tx - 10, y1 + 4, tx - 12, y1);
    }
    cairo_arc(c, x0 + r, y1 - r, r, 0.5 * M_PI, M_PI);
    cairo_close_path(c);
    cairo_path_t *outline = cairo_copy_path(c);

    cairo_save(c);                  /* soft shadow */
    cairo_translate(c, 2, 3);
    cairo_set_source_rgba(c, 0.05, 0.03, 0.02, 0.30);
    cairo_fill(c);
    cairo_restore(c);
    cairo_new_path(c);
    cairo_append_path(c, outline);
    if (who)
        cairo_set_source_rgb(c, 1.0, 0.96, 0.82);      /* the duck's bubbles are warmer */
    else
        cairo_set_source_rgb(c, 1.0, 0.995, 0.98);
    cairo_fill_preserve(c);
    set_rgba(c, INK, 1);
    cairo_set_line_width(c, 2.6);
    cairo_set_line_join(c, CAIRO_LINE_JOIN_ROUND);
    cairo_stroke(c);
    cairo_path_destroy(outline);

    cairo_select_font_face(c, FONT, who ? CAIRO_FONT_SLANT_ITALIC : CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, size);
    cairo_font_extents_t fe;
    cairo_font_extents(c, &fe);
    double block = nl * size * BUB_LINE - (BUB_LINE - 1) * size;
    double ty = y0 + (bh - block) / 2 + fe.ascent - (fe.ascent + fe.descent - size) / 2;
    if (who)
        cairo_set_source_rgb(c, 0.36, 0.20, 0.08);
    else
        set_rgba(c, INK, 1);
    for (int i = 0; i < nl; i++) {
        double lw = text_w(c, lines[i]);
        cairo_move_to(c, bx + (bw - lw) / 2, ty + i * size * BUB_LINE);
        cairo_show_text(c, lines[i]);
    }
    cairo_destroy(c);
    b->born = t;
    b->dying = -1;
}

static void retire(bubble *cur, bubble *old, double t)
{
    if (!cur->surf || cur->dying >= 0)
        return;
    if (old->surf)
        cairo_surface_destroy(old->surf);
    *old = *cur;
    old->dying = t;
    cur->surf = NULL;
}

static void say(const view_t *v, topic_t k, double t)
{
    char txt[160];
    const line_t *l = &LINES[k].l[line_idx[k] % LINES[k].n];
    line_idx[k]++;
    format_line(txt, sizeof(txt), l, v);
    retire(&gpu_bub[0], &gpu_bub[1], t);
    retire(&duck_bub[0], &duck_bub[1], t);
    make_bubble(&gpu_bub[0], txt, 0, t + 0.2);
    cur_topic = k;
    topic_shown[k] = t;
    line_t0 = t;
    pose_t np = pose_for(k, v->anx);
    if (np != pose) {
        pose_prev = pose;
        pose = np;
        pose_t0 = t;
    }
    /* the therapist replies to about every other line */
    int reply = scripted ? (reply_count++ % 2 == 0) : (reply_count++ % 2 == 0 || frand() < 0.25);
    reply_at = reply ? t + REPLY_AT : -1;
}

static void therapist_reply(const view_t *v, double t)
{
    char txt[160];
    const line_t *l;
    if ((reply_idx[cur_topic] + any_idx) % 2 == 0) {       /* alternate: about the topic, a classic */
        l = &LINES[cur_topic].r[reply_idx[cur_topic] % LINES[cur_topic].nr];
        reply_idx[cur_topic]++;
    } else {
        l = &R_ANY[any_idx % N_OF(R_ANY)];
        any_idx++;
    }
    format_line(txt, sizeof(txt), l, v);
    make_bubble(&duck_bub[0], txt, 1, t);
    reply_end = t + REPLY_LEN;
}

/* Back to the start of a session (bench replays the showcase several times) */
static void reset_session(void)
{
    for (int i = 0; i < 2; i++)
        for (bubble *b = &gpu_bub[i]; b; b = b == &gpu_bub[i] ? &duck_bub[i] : NULL) {
            if (b->surf)
                cairo_surface_destroy(b->surf);
            memset(b, 0, sizeof(*b));
        }
    memset(topic_on, 0, sizeof(topic_on));
    memset(topic_since, 0, sizeof(topic_since));
    memset(topic_shown, 0, sizeof(topic_shown));
    memset(line_idx, 0, sizeof(line_idx));
    memset(reply_idx, 0, sizeof(reply_idx));
    memset(parts, 0, sizeof(parts));
    any_idx = reply_count = 0;
    cur_topic = T_IDLE;
    line_t0 = -1e9, reply_at = reply_end = -1;
    pose = pose_prev = POSE_SAD, pose_t0 = -1e9;
    duck_talk_f = ring_phase = fan_ang = page_prog = cab_acc = sweat_acc = steam_acc = 0;
    bags_n = 1;
    memset(bag_t0, 0, sizeof(bag_t0));
    for (int i = 0; i < 5; i++)
        bag_t0[i] = -10;
    bag_gone_t = -10;
}

static void run_session(const view_t *v, double t)
{
    update_topics(v, t);
    double held = t - line_t0;
    if (held >= LINE_MIN) {
        int fresh = -1, oldest = -1;
        for (int k = 0; k < T_IDLE; k++) {
            if (!topic_on[k])
                continue;
            if (fresh < 0 && k != cur_topic && topic_since[k] > topic_shown[k])
                fresh = k;
            if (oldest < 0 || topic_shown[k] < topic_shown[oldest])
                oldest = k;
        }
        if (fresh >= 0)
            say(v, fresh, t);                           /* something new is bothering it */
        else if (!topic_on[cur_topic] || held >= LINE_MAX)
            say(v, oldest >= 0 ? oldest : T_IDLE, t);   /* move on, or keep going */
    }
    if (reply_at >= 0 && t >= reply_at) {
        reply_at = -1;
        therapist_reply(v, t);
    }
    if (duck_bub[0].surf && duck_bub[0].dying < 0 && t >= reply_end)
        retire(&duck_bub[0], &duck_bub[1], t);
}

/* ---------------------------------------------------------------- simulation */

static double log_level(double v, double lo, double hi)
{
    return v <= lo ? 0 : clamp01(log(v / lo) / log(hi / lo));
}

static void ease_view(view_t *v, const stats *g, const sys_stats *s, double dt)
{
    double k = 1 - exp(-dt * 2.5), ks = 1 - exp(-dt * 1.2);
    double busy = gpu_source ? g->tok_s / (N_GPUS * GPU_FULL_RATE) : clamp01(g->tok_s / 1500);
    v->busy   += (busy - v->busy) * k;
    v->tok    += (g->tok_s - v->tok) * k;
    v->power  += (g->power[0] + g->power[1] - v->power) * k;
    v->gtemp  += (fmax(g->temp[0], g->temp[1]) - v->gtemp) * ks;
    v->fan    += (fmax(s->fan[0], s->fan[1]) - v->fan) * ks;
    v->cpu    += (s->cpu - v->cpu) * k;
    v->tctl   += (s->tctl - v->tctl) * ks;
    v->ram    += ((s->ram_total > 0 ? s->ram_used / s->ram_total : 0) - v->ram) * ks;
    v->ram_gb  = s->ram_used;
    v->psi    += (fmax(s->psi_cpu, fmax(s->psi_io, s->psi_mem)) - v->psi) * ks;
    v->ctxt   += (s->ctxt - v->ctxt) * ks;
    v->majflt += (s->majflt - v->majflt) * ks;
    v->swap   += (s->swap - v->swap) * ks;
    double db = s->disk_rd + s->disk_wr, nb = fmax(s->net_rx + s->net_tx, s->ts_rx + s->ts_tx);
    v->diskb  += (db - v->diskb) * k;
    v->netb   += (nb - v->netb) * k;
    v->disk    = log_level(v->diskb, 2e6, 3e9);
    v->net     = log_level(v->netb, 2e5, 1.5e8);
    if (db > 1e5)
        v->disk_rd_frac += (s->disk_rd / db - v->disk_rd_frac) * k;
    /* anxiety: GPU power and CPU temperature */
    double anx = 0.5 * clamp01((v->power - 90) / 900) + 0.5 * clamp01((v->tctl - 50) / 42);
    v->anx    += (anx - v->anx) * ks;
    v->uptime  = s->uptime;
    v->tcp     = s->tcp;
    v->threads = s->n_threads ? s->n_threads : 32;
    v->running = gpu_source ? 0 : g->running;
}

static particle *spawn(ptype type, double x, double y, double vx, double vy, double life)
{
    for (int i = 0; i < MAX_PARTS; i++)
        if (!parts[i].alive) {
            parts[i] = (particle){ type, x, y, vx, vy, 0, 0, 0, life, 0, 1 };
            return &parts[i];
        }
    return NULL;
}

/* Where the patient's forehead is (sweat comes from here), per pose */
static void face_at(double *x, double *y)
{
    *x = GPU_X + (pose == POSE_PANIC ? 44 : 40);
    *y = GPU_Y + (pose == POSE_PANIC ? 10 : 14);
}

/* Luggage: a piece per ~19% of RAM, all five from about 86%; a little hysteresis */
static int bags_for(double ram, int cur)
{
    int n = cur;
    while (n < 5 && ram > 0.19 * n + 0.07)
        n++;
    while (n > 1 && ram < 0.19 * (n - 1) + 0.07 - 0.05)
        n--;
    return n;
}

static void simulate(const view_t *v, double dt, double t)
{
    /* fans spin at the real fan speed (a floor so the art never looks frozen) */
    fan_ang += dt * (0.6 + 9.0 * v->fan);

    /* sweat: flicked off the forehead once anxiety passes a third */
    sweat_acc += dt * fmax(0, v->anx - 0.3) * 9;
    while (sweat_acc >= 1) {
        double fx, fy;
        face_at(&fx, &fy);
        double side = frand() < 0.5 ? -1 : 1;
        spawn(P_SWEAT, fx + side * (6 + frand() * 12), fy + frand() * 6,
              side * (25 + frand() * 40), -60 - frand() * 50, 1.1);
        sweat_acc -= 1;
    }
    /* steam off the card when it is properly hot */
    steam_acc += dt * fmax(0, v->anx - 0.7) * 6;
    while (steam_acc >= 1) {
        particle *p = spawn(P_STEAM, GPU_X + 70 + frand() * 140, GPU_Y + 18, (frand() - 0.5) * 8, -16 - frand() * 10, 2.2);
        if (p)
            p->rot = frand() * 6;
        steam_acc -= 1;
    }

    /* the therapist (the CPU) fills pages as fast as the CPU works; full ones are torn off */
    if (duck_talk_f < 0.5) {
        page_prog += dt * (0.04 + 0.75 * v->cpu);
        if (page_prog >= 1) {
            page_prog = 0;
            particle *p = spawn(P_PAGE, DUCK_X + 38, DUCK_Y + 74, -20 - frand() * 25, -50 - frand() * 20, 9);
            if (p) {
                p->vrot = (frand() - 0.5) * 5;
                p->land = FLOOR_Y - frand() * 8;
            }
        }
    }

    /* the filing cabinet: disk reads throw papers out, writes file them away */
    cab_acc += dt * (v->disk > 0.12 ? 1 + 9 * v->disk : 0);
    while (cab_acc >= 1) {
        int reading = frand() < v->disk_rd_frac;
        particle *p = reading
            ? spawn(P_PAPER, CAB_X + 12 + frand() * 20, CAB_Y + 6, (frand() - 0.6) * 60, -90 - frand() * 50, 1.2)
            : spawn(P_PAPER, CAB_X + 10 + frand() * 24 + 30, CAB_Y - 40 - frand() * 20, -30, 30, 0.9);
        if (p)
            p->vrot = (frand() - 0.5) * 8, p->land = reading ? 1 : -1;
        cab_acc -= 1;
    }

    /* the phone rings in bursts; faster the more traffic there is */
    if (v->net > 0.3)
        ring_phase += dt / (2.6 - 1.4 * v->net);
    else if (fmod(ring_phase, 1.0) > 0.001)
        ring_phase += dt / 1.2;                     /* finish the ring politely */

    int nb = bags_for(v->ram, bags_n);
    if (nb != bags_n) {
        if (nb > bags_n)
            bag_t0[nb - 1] = t;                     /* the new piece drops in */
        else
            bag_gone_t = t;                         /* the old top piece floats off */
        bags_n = nb;
    }

    for (int i = 0; i < MAX_PARTS; i++) {
        particle *p = &parts[i];
        if (!p->alive)
            continue;
        p->age += dt;
        if (p->age >= p->life) {
            p->alive = 0;
            continue;
        }
        switch (p->type) {
        case P_SWEAT:
            p->vy += 320 * dt;
            break;
        case P_PAGE:                                /* flutters down to the floor, then lies there */
            if (p->y < p->land) {
                p->vy = fmin(p->vy + 90 * dt, 38);
                p->vx += sin(p->age * 5) * 50 * dt;
                p->rot += p->vrot * dt;
            } else {
                p->vx = p->vy = 0;
                p->y = p->land;
            }
            break;
        case P_PAPER:
            if (p->land > 0)
                p->vy += 200 * dt;
            p->rot += p->vrot * dt;
            break;
        case P_STEAM:
            p->vx += sin(p->age * 3 + p->rot) * 6 * dt;
            break;
        }
        p->x += p->vx * dt;
        p->y += p->vy * dt;
    }

    double tf = t < reply_end && duck_bub[0].surf ? 1 : 0;
    duck_talk_f += (tf - duck_talk_f) * fmin(1, dt * 7);
}

/* ---------------------------------------------------------------- render */

static void blit(cairo_t *cr, cairo_surface_t *s, double x, double y, double a)
{
    if (a <= 0.003)
        return;
    cairo_set_source_surface(cr, s, x, y);
    if (a >= 0.997)
        cairo_paint(cr);
    else
        cairo_paint_with_alpha(cr, a);
}

static void draw_clock(cairo_t *cr)
{
    time_t now = time(NULL);
    struct tm tm;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&now, &tm);
    double frac = ts.tv_nsec / 1e9;
    double tick = tm.tm_sec + (frac < 0.12 ? sin(frac / 0.12 * M_PI) * 0.12 - 1 + frac / 0.12 : 0);  /* tick with a tiny kick */
    double sec = tm.tm_sec == 0 && frac < 0.12 ? frac / 0.12 * 1 - 1 + 60 : tick;
    double mn = tm.tm_min + tm.tm_sec / 60.0, hr = (tm.tm_hour % 12) + mn / 60.0;
    double r = spr_w(clock_spr) / 2 * 0.78;
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    struct { double ang, len, w; rgb c; } hands[3] = {
        { hr / 12, 0.52, 3.2, INK }, { mn / 60, 0.80, 2.2, INK }, { sec / 60, 0.86, 1.1, { 0.78, 0.16, 0.10 } },
    };
    for (int i = 0; i < 3; i++) {
        double a = hands[i].ang * 2 * M_PI - M_PI / 2;
        set_rgb(cr, hands[i].c);
        cairo_set_line_width(cr, hands[i].w);
        cairo_move_to(cr, CLOCK_X - cos(a) * r * 0.14, CLOCK_Y - sin(a) * r * 0.14);
        cairo_line_to(cr, CLOCK_X + cos(a) * r * hands[i].len, CLOCK_Y + sin(a) * r * hands[i].len);
        cairo_stroke(cr);
    }
    cairo_arc(cr, CLOCK_X, CLOCK_Y, 2.2, 0, 2 * M_PI);
    set_rgb(cr, BRASS);
    cairo_fill(cr);
}

/* PSI needle: 0 at the left of the dial, 40% stalled pegs it in the red */
static void draw_gauge(cairo_t *cr, const view_t *v, double t)
{
    double f = clamp01(v->psi / 40);
    double a = (150 + 240 * f) * M_PI / 180 + sin(t * 23) * 0.04 * f * f;
    double r = spr_w(gauge_spr) / 2 * 0.62;
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(cr, 2.2);
    cairo_set_source_rgb(cr, 0.70, 0.12, 0.08);
    cairo_move_to(cr, GAUGE_X - cos(a) * 3, GAUGE_Y - sin(a) * 3);
    cairo_line_to(cr, GAUGE_X + cos(a) * r, GAUGE_Y + sin(a) * r);
    cairo_stroke(cr);
    cairo_arc(cr, GAUGE_X, GAUGE_Y, 2.6, 0, 2 * M_PI);
    set_rgb(cr, BRASS);
    cairo_fill(cr);
}

/* Thermometer fluid: 30..100 C */
static void draw_thermo(cairo_t *cr, const view_t *v)
{
    double f = clamp01((v->tctl - 30) / 70);
    double top = THERMO_Y + THERMO_H - 12, bot = THERMO_Y + 8;
    double y = top - (top - bot) * f;
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(cr, 2.4);
    cairo_set_source_rgb(cr, 0.86, 0.16, 0.10);
    cairo_move_to(cr, THERMO_X, THERMO_Y + THERMO_H - 9);
    cairo_line_to(cr, THERMO_X, y);
    cairo_stroke(cr);
}

static void draw_cabinet(cairo_t *cr, const view_t *v, double t)
{
    double jx = v->disk > 0.4 ? sin(t * 53) * (v->disk - 0.4) * 2.2 : 0;
    blit(cr, cab_spr, CAB_X + jx, CAB_Y, 1);
}

static void draw_bags(cairo_t *cr, double t)
{
    double y = BAGS_Y;
    double gone = t - bag_gone_t;
    int n = bags_n < 5 && gone < 0.5 ? bags_n + 1 : bags_n;
    for (int i = 0; i < n; i++) {
        cairo_surface_t *s = bag_spr[i];
        double w = spr_w(s), h = spr_h(s);
        double drop = 0, a = 1;
        double age = t - bag_t0[i];
        if (i >= bags_n) {                          /* letting go: rises and fades */
            drop = -30 * gone;
            a = 1 - gone / 0.5;
        } else if (age >= 0 && age < 0.6) {         /* falls in with a small bounce */
            double u = age / 0.6;
            drop = -60 * (1 - u) * (1 - u) + (u > 0.7 ? -4 * sin((u - 0.7) / 0.3 * M_PI) : 0);
            a = fmin(1, u * 3);
        }
        y = i == 0 ? BAGS_Y - h : y - h * 0.86;     /* pieces sit a little into the one below */
        blit(cr, s, BAGS_X - w / 2, y + drop, a);
    }
}

static void draw_phone(cairo_t *cr, const view_t *v, double t)
{
    double w = spr_w(phone_spr);
    double ph = fmod(ring_phase, 1.0), ringing = ring_phase > 0 && ph < 0.55 && ph > 0.001 ? 1 : 0;
    double jx = ringing ? sin(t * 70) * 1.6 : 0, jy = ringing ? -fabs(sin(t * 35)) * 1.6 : 0;
    /* table */
    cairo_save(cr);
    cairo_rectangle(cr, PHONE_X, PHONE_Y + PHONE_SPLIT, w, spr_h(phone_spr) - PHONE_SPLIT);
    cairo_clip(cr);
    blit(cr, phone_spr, PHONE_X, PHONE_Y, 1);
    cairo_restore(cr);
    /* phone, rattling while it rings */
    cairo_save(cr);
    cairo_rectangle(cr, PHONE_X - 4, PHONE_Y - 4, w + 8, PHONE_SPLIT + 4);
    cairo_clip(cr);
    blit(cr, phone_spr, PHONE_X + jx, PHONE_Y + jy, 1);
    cairo_restore(cr);
    if (ringing) {                                  /* ring lines either side */
        double cx = PHONE_X + w * 0.55, cy = PHONE_Y + 10, a = 0.55 + 0.45 * v->net;
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(cr, 2.2);
        for (int side = -1; side <= 1; side += 2)
            for (int k = 0; k < 2; k++) {
                double rr = 22 + k * 7 + 2 * sin(t * 18);
                double a0 = side < 0 ? M_PI - 0.5 : -0.5, a1 = side < 0 ? M_PI + 0.5 : 0.5;
                cairo_new_path(cr);
                cairo_arc(cr, cx, cy, rr, a0, a1);
                set_rgba(cr, INK, a * (1 - 0.35 * k));
                cairo_stroke(cr);
            }
    }
}

/* The patient: pose cross-fade, heat flush, trembling, and its three fans spinning */
static const double FAN_SRC[N_POSE][3][2] = {
    { { 0, 0 }, { 821, 528 }, { 1083, 544 } },             /* sad: an arm covers the first fan */
    { { 0, 0 }, { 823, 525 }, { 1083, 544 } },             /* talk: a hand covers the first fan */
    { { 559, 564 }, { 812, 582 }, { 1081, 612 } },
};
#define FAN_R_SRC       112.0

static void draw_pose(cairo_t *cr, pose_t p, double x, double y, double a, double heat)
{
    blit(cr, gpu_spr[p], x, y, a);
    blit(cr, gpu_hot[p], x, y, a * heat);
    for (int f = 0; f < 3; f++) {
        if (FAN_SRC[p][f][0] == 0)
            continue;
        double fx = x + (FAN_SRC[p][f][0] - GPU_SRC_X) * GPU_K, fy = y + (FAN_SRC[p][f][1] - GPU_SRC_Y) * GPU_K;
        cairo_save(cr);
        cairo_arc(cr, fx, fy, FAN_R_SRC * GPU_K, 0, 2 * M_PI);
        cairo_clip(cr);
        cairo_translate(cr, fx, fy);
        cairo_rotate(cr, fan_ang + f * 0.9);
        cairo_translate(cr, -fx, -fy);
        blit(cr, gpu_spr[p], x, y, a);
        blit(cr, gpu_hot[p], x, y, a * heat);
        cairo_restore(cr);
    }
}

static void draw_gpu(cairo_t *cr, const view_t *v, double t)
{
    double shake = fmax(0, v->anx - 0.35) / 0.65;
    double jx = shake * shake * 2.4 * (0.6 * sin(t * 37) + 0.4 * sin(t * 59 + 1));
    double jy = shake * shake * 1.2 * sin(t * 43 + 2);
    double breathe = sin(t * (1.2 + 3 * v->anx)) * (0.6 + 0.6 * v->anx);
    double x = GPU_X + jx, y = GPU_Y + jy + breathe;
    double heat = clamp01((v->anx - 0.42) / 0.5);
    double f = clamp01((t - pose_t0) / 0.3);
    if (f < 1)
        draw_pose(cr, pose_prev, x, y, 1 - f, heat);
    draw_pose(cr, pose, x, y, f, heat);

    /* worry lines by the head when anxious */
    if (v->anx > 0.5) {
        double fx, fy, a = clamp01((v->anx - 0.5) / 0.3) * (0.7 + 0.3 * sin(t * 9));
        face_at(&fx, &fy);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(cr, 2.2);
        set_rgba(cr, INK, a);
        for (int i = 0; i < 3; i++) {
            double ang = -M_PI / 2 - 0.75 + i * 0.75;
            cairo_move_to(cr, fx + jx + cos(ang) * 30, fy + jy + sin(ang) * 26);
            cairo_line_to(cr, fx + jx + cos(ang) * 40, fy + jy + sin(ang) * 35);
            cairo_stroke(cr);
        }
    }
}

/* The therapist: pose cross-fade, and the notes being scribbled on the clipboard */
static void draw_duck(cairo_t *cr, const view_t *v, double t)
{
    double f = duck_talk_f;
    blit(cr, duck_spr[0], DUCK_X, DUCK_Y, 1 - f);
    blit(cr, duck_spr[1], DUCK_X, DUCK_Y, f);
    if (f > 0.5)
        return;
    /* pen strokes: faster with CPU load */
    double a = 1 - f * 2, sp = 3 + 22 * v->cpu;
    double px = DUCK_X + 38 + sin(t * sp) * 2.5, py = DUCK_Y + 58 + sin(t * sp * 0.5) * 1.2;
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(cr, 2.0);
    for (int i = 0; i < 3; i++) {                   /* speed marks behind the pen */
        double ph = fmod(t * sp * 0.25 + i / 3.0, 1.0);
        double al = a * clamp01(v->cpu * 2.5) * sin(ph * M_PI);
        set_rgba(cr, INK, al * 0.9);
        cairo_new_path(cr);
        cairo_move_to(cr, px - 4 - i * 3, py - 16 + i * 5);
        cairo_line_to(cr, px - 11 - i * 3 - 4 * ph, py - 18 + i * 5);
        cairo_stroke(cr);
    }
}

static void draw_particles(cairo_t *cr, double t)
{
    (void)t;
    for (int i = 0; i < MAX_PARTS; i++) {
        const particle *p = &parts[i];
        if (!p->alive)
            continue;
        double u = p->age / p->life;
        switch (p->type) {
        case P_SWEAT: {                             /* cartoon drop pointing the way it flies */
            double a = u < 0.75 ? 1 : (1 - u) / 0.25;
            double ang = atan2(p->vy, p->vx) - M_PI / 2;
            cairo_save(cr);
            cairo_translate(cr, p->x, p->y);
            cairo_rotate(cr, ang + M_PI);
            cairo_scale(cr, 1.35, 1.35);
            cairo_new_path(cr);
            cairo_move_to(cr, 0, -7);
            cairo_curve_to(cr, 2, -3, 4.2, 0, 4.2, 2);
            cairo_arc(cr, 0, 2, 4.2, 0, M_PI);
            cairo_curve_to(cr, -4.2, 0, -2, -3, 0, -7);
            cairo_close_path(cr);
            cairo_set_source_rgba(cr, 0.55, 0.82, 1.0, a);
            cairo_fill_preserve(cr);
            cairo_set_line_width(cr, 1.3);
            set_rgba(cr, INK, a * 0.9);
            cairo_stroke(cr);
            cairo_arc(cr, -1.4, 1.5, 1.2, 0, 2 * M_PI);
            cairo_set_source_rgba(cr, 1, 1, 1, a * 0.9);
            cairo_fill(cr);
            cairo_restore(cr);
            break;
        }
        case P_PAGE:
        case P_PAPER: {
            double a = p->type == P_PAGE ? (u < 0.85 ? 1 : (1 - u) / 0.15)
                                         : (u < 0.6 ? 1 : (1 - u) / 0.4);
            double w = p->type == P_PAGE ? 11 : 9, h = p->type == P_PAGE ? 14 : 11;
            cairo_save(cr);
            cairo_translate(cr, p->x, p->y);
            cairo_rotate(cr, p->rot);
            cairo_scale(cr, 1, p->type == P_PAGE && p->y < p->land ? 0.55 + 0.45 * fabs(sin(p->age * 4)) : 1);
            cairo_rectangle(cr, -w / 2, -h / 2, w, h);
            set_rgba(cr, PAPER, a);
            cairo_fill_preserve(cr);
            cairo_set_line_width(cr, 1.1);
            set_rgba(cr, INK, a * 0.8);
            cairo_stroke(cr);
            cairo_set_line_width(cr, 0.9);
            cairo_set_source_rgba(cr, 0.25, 0.35, 0.65, a * 0.7);
            for (int l = 0; l < 3; l++) {           /* scribbled notes */
                cairo_move_to(cr, -w / 2 + 2, -h / 2 + 3.5 + l * 3.2);
                cairo_line_to(cr, w / 2 - 2 - (l == 2) * 3, -h / 2 + 3.5 + l * 3.2);
            }
            cairo_stroke(cr);
            cairo_restore(cr);
            break;
        }
        case P_STEAM: {                             /* cartoon heat wiggles rising off the card */
            double a = sin(u * M_PI) * 0.75;
            cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
            cairo_set_line_width(cr, 2.0);
            cairo_new_path(cr);
            for (int k = 0; k <= 8; k++) {
                double yy = p->y - k * 2.2, xx = p->x + sin(k * 0.9 + p->age * 6 + p->rot) * 2.6;
                if (k == 0)
                    cairo_move_to(cr, xx, yy);
                else
                    cairo_line_to(cr, xx, yy);
            }
            cairo_set_source_rgba(cr, 1, 0.96, 0.9, a);
            cairo_stroke(cr);
            break;
        }
        }
    }
}

static void draw_bubble(cairo_t *cr, const bubble *b, double t)
{
    if (!b->surf)
        return;
    double a, s;
    if (b->dying >= 0) {                            /* shrink away */
        double u = clamp01((t - b->dying) / 0.18);
        a = 1 - u;
        s = 1 - 0.08 * u;
    } else {                                        /* pop in with a little overshoot */
        double u = clamp01((t - b->born) / 0.28);
        if (t < b->born)
            return;
        a = clamp01(u * 2.5);
        s = u >= 1 ? 1 : 1 + 2.2 * pow(u - 1, 3) + 1.2 * pow(u - 1, 2);
        s = 0.6 + 0.4 * s;
    }
    if (a <= 0)
        return;
    cairo_save(cr);
    if (fabs(s - 1) > 0.002) {
        cairo_translate(cr, b->ax, b->ay);
        cairo_scale(cr, s, s);
        cairo_translate(cr, -b->ax, -b->ay);
    }
    blit(cr, b->surf, b->x, b->y, a);
    cairo_restore(cr);
}

/* The notepad at the bottom: tok/s (or GPU %) in ink, redrawn at most 4x a second */
static void update_hud(const view_t *v, double t)
{
    static double shown, next;
    char key[64], num[32];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        shown = shown_rate(v->tok);
    }
    snprintf(num, sizeof(num), "%.0f", shown);
    snprintf(key, sizeof(key), "%s|%d", num, gpu_source);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);

    const double w = 158, h = 50, oy = CARD_Y - 45;         /* surface covers y oy..oy+90 */
    cairo_t *c = cairo_create(hud_cache);
    cairo_set_operator(c, CAIRO_OPERATOR_CLEAR);
    cairo_paint(c);
    cairo_set_operator(c, CAIRO_OPERATOR_OVER);
    cairo_translate(c, CARD_X, CARD_Y - oy);
    cairo_rotate(c, -0.035);
    /* shadow, paper, ruled lines, margin, spiral */
    cairo_rectangle(c, -w / 2 + 3, -h / 2 + 4, w, h);
    cairo_set_source_rgba(c, 0.05, 0.03, 0.02, 0.35);
    cairo_fill(c);
    cairo_rectangle(c, -w / 2, -h / 2, w, h);
    set_rgb(c, PAPER);
    cairo_fill_preserve(c);
    set_rgba(c, INK, 0.9);
    cairo_set_line_width(c, 1.6);
    cairo_stroke(c);
    cairo_set_line_width(c, 1);
    cairo_set_source_rgba(c, 0.35, 0.55, 0.85, 0.45);
    for (int i = 1; i < 4; i++) {
        cairo_move_to(c, -w / 2 + 2, -h / 2 + 6 + i * 11.5);
        cairo_line_to(c, w / 2 - 2, -h / 2 + 6 + i * 11.5);
    }
    cairo_stroke(c);
    cairo_set_source_rgba(c, 0.85, 0.25, 0.2, 0.55);
    cairo_move_to(c, -w / 2 + 14, -h / 2 + 1);
    cairo_line_to(c, -w / 2 + 14, h / 2 - 1);
    cairo_stroke(c);
    set_rgba(c, INK, 0.85);
    cairo_set_line_width(c, 1.6);
    for (int i = 0; i < 9; i++) {
        double sx = -w / 2 + 14 + i * 16.5;
        cairo_new_path(c);
        cairo_arc(c, sx, -h / 2, 3.2, M_PI * 0.9, M_PI * 2.1);
        cairo_stroke(c);
    }

    /* number + unit, centred together */
    const char *unit = gpu_source ? "% GPU" : " tok/s";
    cairo_text_extents_t en, eu;
    cairo_select_font_face(c, FONT, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, 32);
    cairo_text_extents(c, num, &en);
    cairo_set_font_size(c, 22);
    cairo_text_extents(c, unit, &eu);
    double tw = en.x_advance + eu.x_advance, x0 = -tw / 2 + 6, base = 12;
    cairo_set_source_rgb(c, 0.13, 0.20, 0.45);          /* blue ink */
    cairo_set_font_size(c, 32);
    cairo_move_to(c, x0, base);
    cairo_show_text(c, num);
    cairo_set_font_size(c, 22);
    cairo_move_to(c, x0 + en.x_advance, base);
    cairo_show_text(c, unit);
    cairo_destroy(c);
}

static void render(cairo_t *cr, const view_t *v, double t)
{
    cairo_set_source_surface(cr, bg_cache, 0, 0);
    cairo_paint(cr);
    draw_clock(cr);
    draw_gauge(cr, v, t);
    draw_thermo(cr, v);
    draw_cabinet(cr, v, t);
    draw_duck(cr, v, t);
    draw_gpu(cr, v, t);
    draw_phone(cr, v, t);
    draw_bags(cr, t);
    draw_particles(cr, t);
    update_hud(v, t);
    cairo_set_source_surface(cr, hud_cache, 0, CARD_Y - 45);
    cairo_paint(cr);
    draw_bubble(cr, &duck_bub[1], t);
    draw_bubble(cr, &gpu_bub[1], t);
    draw_bubble(cr, &gpu_bub[0], t);
    draw_bubble(cr, &duck_bub[0], t);
}

/* ---------------------------------------------------------------- main */

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
    scripted = showcase || bench;

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    srand(showcase || bench ? 7 : (unsigned)time(NULL));
    if (!scripted)                                  /* start somewhere different each run */
        for (int k = 0; k < N_TOPIC; k++)
            line_idx[k] = rand() % LINES[k].n, reply_idx[k] = rand() % LINES[k].nr;
    load_assets();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* showcase moments: overheating, busy, sulking; the whole session is replayed up to each */
        struct { double at; const char *png; } scenes[] = {
            { 39.0, "therapy_preview.png" }, { 15.5, "therapy_busy.png" }, { 3.5, "therapy_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            size_t len = 0;
            double b0 = 0;
            int n = (int)(scenes[k].at * FPS_BUSY);
            memset(&v, 0, sizeof(v));
            reset_session();
            for (int i = 0; i < n; i++) {
                double t = i / (double)FPS_BUSY;
                if (i == n - 60)
                    b0 = now_s();
                showcase_poll(&g, &s, t);
                ease_view(&v, &g, &s, 1.0 / FPS_BUSY);
                run_session(&v, t);
                simulate(&v, 1.0 / FPS_BUSY, t);
                render(cr, &v, t);
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
    while (!stop) {
        double t = now_s(), dt = t - last;
        last = t;
        if (dt > 0.25)
            dt = 0.25;

        if (showcase) {
            showcase_poll(&g, &s, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + POLL_S;
            if (demo) {
                /* fresh values each poll from the clock; nothing is scaled in place */
                static stats demo_base;
                demo_poll(&demo_base, t - t0);
                g = demo_base;
                g.power[0] = 30 + 520 * g.load[0];
                g.power[1] = 34 + 510 * g.load[1];
                g.tok_port[0] *= 5, g.tok_port[1] *= 5;
                g.tok_s = g.tok_port[0] + g.tok_port[1];
                if (g.tok_s < 120)
                    g.tok_s = g.tok_port[0] = g.tok_port[1] = 0, g.running = 0;
                if (gpu_source)
                    gpu_rate_poll(&g);
                sys_demo(&s, t - t0);
            } else {
                gpus_poll(&g);
                gpu_fans_poll(s.fan);
                if (gpu_source)
                    gpu_rate_poll(&g);
                else
                    vllm_poll(&g, t);
                sys_poll(&s, t);
            }
        }

        ease_view(&v, &g, &s, dt);
        run_session(&v, t - t0);
        simulate(&v, dt, t - t0);
        render(cr, &v, t - t0);
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

        /* a calm, sulking patient gets the idle frame rate */
        int idle = cur_topic == T_IDLE && v.anx < 0.3 && v.net < 0.3 && v.disk < 0.12 && !showcase;
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
    cairo_surface_destroy(hud_cache);
    for (int i = 0; i < N_POSE; i++) {
        cairo_surface_destroy(gpu_spr[i]);
        cairo_surface_destroy(gpu_hot[i]);
    }
    for (int i = 0; i < 2; i++) {
        cairo_surface_destroy(duck_spr[i]);
        for (bubble *b = &gpu_bub[i]; b; b = b == &gpu_bub[i] ? &duck_bub[i] : NULL)
            if (b->surf)
                cairo_surface_destroy(b->surf);
    }
    for (int i = 0; i < 5; i++)
        cairo_surface_destroy(bag_spr[i]);
    cairo_surface_destroy(clock_spr);
    cairo_surface_destroy(gauge_spr);
    cairo_surface_destroy(phone_spr);
    cairo_surface_destroy(cab_spr);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
