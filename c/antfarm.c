/*
 * antfarm: a round glass ant farm on the iCUE LINK AIO pump LCD, where the colony is
 * your whole machine.
 *
 * A backlit cross-section of soil with tunnels and chambers. Worker ants are CPU work:
 * how many are out and how fast they walk follow total CPU load, and 16 brood cells
 * (one per core, both SMT threads) glow and bustle with that core's load; CCD0 is the
 * left half of the colony, CCD1 the right. The big food store fills with seeds as RAM
 * fills (pale crumbs on top are page cache), and ants carry seeds between it and a deep
 * cellar when the machine swaps. Three chambers are the NVMe drives: amber crumbs go
 * down for writes, pale ones come up for reads. Ants on the surface bring leaves in
 * (network in) and head out (network out), and the burrows along the surface are the
 * open TCP connections. The queen pulses with GPU activity and lays an egg for bursts
 * of new processes; eggs hatch into workers. The two fungus nurseries glow with each
 * GPU's power (blue GPU 0, amber GPU 1). Pressure stalls (PSI) jam the tunnels. The
 * soil warms with CPU temperature, and when it runs hot the ants wave their antennae
 * and scurry. The plaque on the stand counts every task on the machine.
 * Everything is drawn procedurally with cairo; static layers are cached.
 * Run with --demo to simulate data, --showcase for a scripted 36 s arc,
 * --bench to write preview PNGs. ANTFARM_DEBUG=1 prints the sensor readings.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <cairo/cairo.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/hidraw.h>
#include <limits.h>
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


/* ---------------------------------------------------------------- system sensors */
/* ---- system sensors ---- */

/*
 * Reusable machine sensors, polled at most ~2 Hz. hwmon devices are found by name at
 * startup (never by number); rates are differenced between polls.
 */
#define SYS_MAX_CPU     64
#define SYS_NVME        3

typedef struct {
    int    ncpu;                    /* threads seen in /proc/stat */
    double cpu_total;               /* 0..1 */
    double cpu[SYS_MAX_CPU];        /* per thread 0..1 */
    double cpu_temp;                /* k10temp Tctl, C */
    double board_temp[5];           /* asusec: CPU, CPU Package, Motherboard, T_Sensor, VRM (C) */
    double fan_rpm;                 /* asusec fan1 */
    double mem_total, mem_used, mem_cached;     /* GB; used = total - available */
    double dimm_temp[2];            /* spd5118, C */
    double nvme_temp[SYS_NVME];     /* C, indexed like nvmeN */
    double disk_rd[SYS_NVME], disk_wr[SYS_NVME];/* bytes/s for nvmeNn1 */
    double net_rx, net_tx;          /* enp12s0 bytes/s */
    double ts_rx, ts_tx;            /* tailscale0 bytes/s */
    double forks_s, ctxt_s, intr_s; /* /proc/stat: new processes, context switches, interrupts per second */
    int    procs_running, tasks;    /* runnable now; all tasks (threads) from /proc/loadavg */
    double load1;
    double psi_cpu, psi_io, psi_mem;/* PSI "some": % of time stalled over the last poll */
    double swapin_s, swapout_s;     /* pages/s (zram) */
    double pgfault_s, pgmajfault_s;
    int    tcp_inuse;               /* open TCP sockets */
    double cpu_watts;               /* package power from RAPL, -1 if unreadable (needs root) */
} sys_stats;

static char sys_k10[320], sys_asus[320], sys_dimm[2][320], sys_nvme_hw[SYS_NVME][320];

static double read_num(const char *path, double fail)
{
    char buf[64];
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return fail;
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0)
        return fail;
    buf[n] = 0;
    return strtod(buf, NULL);
}

static void sys_init(void)
{
    DIR *dir = opendir("/sys/class/hwmon");
    struct dirent *e;
    int ndimm = 0;
    if (!dir)
        return;
    while ((e = readdir(dir))) {
        char path[300], name[64] = "", link[PATH_MAX];
        if (strncmp(e->d_name, "hwmon", 5) != 0)
            continue;
        snprintf(path, sizeof(path), "/sys/class/hwmon/%s/name", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        if (!fgets(name, sizeof(name), f))
            name[0] = 0;
        fclose(f);
        name[strcspn(name, "\n")] = 0;
        snprintf(path, sizeof(path), "/sys/class/hwmon/%s", e->d_name);
        if (!strcmp(name, "k10temp"))
            snprintf(sys_k10, sizeof(sys_k10), "%s", path);
        else if (!strcmp(name, "asusec"))
            snprintf(sys_asus, sizeof(sys_asus), "%s", path);
        else if (!strcmp(name, "spd5118") && ndimm < 2)
            snprintf(sys_dimm[ndimm++], sizeof(sys_dimm[0]), "%s", path);
        else if (!strcmp(name, "nvme")) {
            /* which controller: the device link ends in .../nvme/nvmeN */
            char dev[320];
            snprintf(dev, sizeof(dev), "%s/device", path);
            if (realpath(dev, link)) {
                const char *b = strrchr(link, '/');
                int k = b ? atoi(b + 5) : -1;
                if (b && !strncmp(b + 1, "nvme", 4) && k >= 0 && k < SYS_NVME)
                    snprintf(sys_nvme_hw[k], sizeof(sys_nvme_hw[0]), "%s", path);
            }
        }
    }
    closedir(dir);
}

/* first "total=" in a /proc/pressure file (microseconds stalled, "some" line) */
static unsigned long long psi_total(const char *path)
{
    char buf[256];
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return 0;
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0)
        return 0;
    buf[n] = 0;
    const char *p = strstr(buf, "total=");
    return p ? strtoull(p + 6, NULL, 10) : 0;
}

static void sys_poll(sys_stats *s, double t)
{
    static unsigned long long cbusy[SYS_MAX_CPU + 1], ctot[SYS_MAX_CPU + 1];
    static unsigned long long dsec_r[SYS_NVME], dsec_w[SYS_NVME], nrx[2], ntx[2];
    static unsigned long long forks, ctxt, intr, psi[3], swin, swout, pgf, pgmaj, rapl;
    static double last_t;
    static int have, rapl_ok = 1;
    static char statbuf[65536];
    char line[512], path[1024];
    double dt = t - last_t;
    FILE *f;
#define RATE(now, prev) (have && dt > 0 && (now) >= (prev) ? ((now) - (prev)) / dt : 0)

    /* /proc/stat: per-thread busy time and the kernel's counters, differenced */
    int sfd = open("/proc/stat", O_RDONLY | O_CLOEXEC);
    if (sfd >= 0) {
        ssize_t len = 0, r;
        while (len < (ssize_t)sizeof(statbuf) - 1 && (r = read(sfd, statbuf + len, sizeof(statbuf) - 1 - len)) > 0)
            len += r;
        close(sfd);
        statbuf[len] = 0;
        int n = 0;
        char *save = NULL;
        for (char *ln = strtok_r(statbuf, "\n", &save); ln; ln = strtok_r(NULL, "\n", &save)) {
            unsigned long long v;
            if (sscanf(ln, "ctxt %llu", &v) == 1) { s->ctxt_s = RATE(v, ctxt); ctxt = v; continue; }
            if (sscanf(ln, "intr %llu", &v) == 1) { s->intr_s = RATE(v, intr); intr = v; continue; }
            if (sscanf(ln, "processes %llu", &v) == 1) { s->forks_s = RATE(v, forks); forks = v; continue; }
            if (sscanf(ln, "procs_running %llu", &v) == 1) { s->procs_running = (int)v; continue; }
            if (strncmp(ln, "cpu", 3))
                continue;
            unsigned long long c[8] = { 0 };
            int idx = ln[3] == ' ' ? SYS_MAX_CPU : atoi(ln + 3);
            char *p = strchr(ln, ' ');
            if (!p || idx < 0 || idx > SYS_MAX_CPU)
                continue;
            sscanf(p, "%llu %llu %llu %llu %llu %llu %llu %llu", &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7]);
            unsigned long long tot = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
            unsigned long long busy = tot - c[3] - c[4];
            if (have && tot > ctot[idx]) {
                double u = (double)(busy - cbusy[idx]) / (double)(tot - ctot[idx]);
                if (idx == SYS_MAX_CPU)
                    s->cpu_total = clamp01(u);
                else
                    s->cpu[idx] = clamp01(u);
            }
            cbusy[idx] = busy;
            ctot[idx] = tot;
            if (idx < SYS_MAX_CPU && idx + 1 > n)
                n = idx + 1;
        }
        s->ncpu = n;
    }

    if ((f = fopen("/proc/loadavg", "r"))) {
        int run, tasks;
        if (fscanf(f, "%lf %*f %*f %d/%d", &s->load1, &run, &tasks) == 3)
            s->tasks = tasks;
        fclose(f);
    }

    /* pressure stall information: share of wall time some task was stalled */
    static const char *psi_path[3] = { "/proc/pressure/cpu", "/proc/pressure/io", "/proc/pressure/memory" };
    double *psi_out[3] = { &s->psi_cpu, &s->psi_io, &s->psi_mem };
    for (int i = 0; i < 3; i++) {
        unsigned long long v = psi_total(psi_path[i]);
        *psi_out[i] = fmin(100, RATE(v, psi[i]) / 1e4);
        psi[i] = v;
    }

    if ((f = fopen("/proc/vmstat", "r"))) {
        int got = 0;
        while (got < 4 && fgets(line, sizeof(line), f)) {
            unsigned long long v;
            if (sscanf(line, "pswpin %llu", &v) == 1) { s->swapin_s = RATE(v, swin); swin = v; got++; }
            else if (sscanf(line, "pswpout %llu", &v) == 1) { s->swapout_s = RATE(v, swout); swout = v; got++; }
            else if (sscanf(line, "pgfault %llu", &v) == 1) { s->pgfault_s = RATE(v, pgf); pgf = v; got++; }
            else if (sscanf(line, "pgmajfault %llu", &v) == 1) { s->pgmajfault_s = RATE(v, pgmaj); pgmaj = v; got++; }
        }
        fclose(f);
    }

    if ((f = fopen("/proc/net/sockstat", "r"))) {
        while (fgets(line, sizeof(line), f))
            if (sscanf(line, "TCP: inuse %d", &s->tcp_inuse) == 1)
                break;
        fclose(f);
    }

    /* CPU package power: root-only on most systems, so give up quietly on EACCES */
    s->cpu_watts = -1;
    if (rapl_ok) {
        int fd = open("/sys/class/powercap/intel-rapl:0/energy_uj", O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            rapl_ok = 0;
        } else {
            char b[32];
            ssize_t n = read(fd, b, sizeof(b) - 1);
            close(fd);
            if (n > 0) {
                b[n] = 0;
                unsigned long long v = strtoull(b, NULL, 10);
                if (have && rapl && v >= rapl && dt > 0)
                    s->cpu_watts = (v - rapl) / 1e6 / dt;
                rapl = v;
            } else {
                rapl_ok = 0;
            }
        }
    }

    /* temperatures */
    if (sys_k10[0]) {
        snprintf(path, sizeof(path), "%s/temp1_input", sys_k10);
        s->cpu_temp = read_num(path, 0) / 1000.0;
    }
    if (sys_asus[0]) {
        for (int i = 0; i < 5; i++) {
            snprintf(path, sizeof(path), "%s/temp%d_input", sys_asus, i + 1);
            s->board_temp[i] = read_num(path, 0) / 1000.0;
        }
        snprintf(path, sizeof(path), "%s/fan1_input", sys_asus);
        s->fan_rpm = read_num(path, 0);
    }
    for (int i = 0; i < 2; i++)
        if (sys_dimm[i][0]) {
            snprintf(path, sizeof(path), "%s/temp1_input", sys_dimm[i]);
            s->dimm_temp[i] = read_num(path, 0) / 1000.0;
        }
    for (int i = 0; i < SYS_NVME; i++)
        if (sys_nvme_hw[i][0]) {
            snprintf(path, sizeof(path), "%s/temp1_input", sys_nvme_hw[i]);
            s->nvme_temp[i] = read_num(path, 0) / 1000.0;
        }

    /* memory */
    if ((f = fopen("/proc/meminfo", "r"))) {
        double total = 0, avail = 0, cached = 0, v;
        int got = 0;
        while (got < 3 && fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %lf", &v) == 1) total = v, got++;
            else if (sscanf(line, "MemAvailable: %lf", &v) == 1) avail = v, got++;
            else if (sscanf(line, "Cached: %lf", &v) == 1) cached = v, got++;
        }
        fclose(f);
        s->mem_total = total / 1048576.0;
        s->mem_used = (total - avail) / 1048576.0;
        s->mem_cached = cached / 1048576.0;
    }

    /* NVMe throughput: sectors read (field 6) and written (field 10), 512 bytes each */
    if ((f = fopen("/proc/diskstats", "r"))) {
        while (fgets(line, sizeof(line), f)) {
            char name[32];
            unsigned long long r, w;
            int k;
            if (sscanf(line, "%*u %*u %31s %*u %*u %llu %*u %*u %*u %llu", name, &r, &w) != 3)
                continue;
            if (strncmp(name, "nvme", 4) || sscanf(name, "nvme%dn1", &k) != 1 || strchr(name, 'p') || k < 0 || k >= SYS_NVME)
                continue;
            if (have && dt > 0) {
                s->disk_rd[k] = r >= dsec_r[k] ? (r - dsec_r[k]) * 512.0 / dt : 0;
                s->disk_wr[k] = w >= dsec_w[k] ? (w - dsec_w[k]) * 512.0 / dt : 0;
            }
            dsec_r[k] = r;
            dsec_w[k] = w;
        }
        fclose(f);
    }

    /* network */
    if ((f = fopen("/proc/net/dev", "r"))) {
        while (fgets(line, sizeof(line), f)) {
            char *c = strchr(line, ':'), *p = line;
            unsigned long long rx, tx;
            int k;
            if (!c)
                continue;
            *c = 0;
            while (*p == ' ')
                p++;
            k = !strcmp(p, "enp12s0") ? 0 : !strcmp(p, "tailscale0") ? 1 : -1;
            if (k < 0 || sscanf(c + 1, "%llu %*u %*u %*u %*u %*u %*u %*u %llu", &rx, &tx) != 2)
                continue;
            if (have && dt > 0) {
                double r = rx >= nrx[k] ? (rx - nrx[k]) / dt : 0, w = tx >= ntx[k] ? (tx - ntx[k]) / dt : 0;
                if (k == 0) s->net_rx = r, s->net_tx = w;
                else        s->ts_rx = r, s->ts_tx = w;
            }
            nrx[k] = rx;
            ntx[k] = tx;
        }
        fclose(f);
    }
    last_t = t;
    have = 1;
#undef RATE
}

/* ---- end system sensors ---- */

/* ---------------------------------------------------------------- scene */

/*
 * The colony is a graph: nodes are chambers and junctions, edges are tunnels (curves
 * resampled every 2 px). Ants walk along edges at sub-pixel positions, their drawn
 * position and heading eased so they never jump or snap, and turn around smoothly in
 * dead ends. Ant sprites are pre-rendered at 64 headings x 4 leg phases x 2 antenna
 * poses and blitted unrotated. The soil is painted twice at startup (cool and hot) and
 * the two are blended into a cached background only when the temperature tint moves a
 * step. Glows are small sprites added over their own area; text is a cached strip.
 */
#define FPS_BUSY        20
#define FPS_IDLE        15
#define CX              240.0
#define CY              240.0
#define R_GLASS         226.0       /* inside edge of the frame */
#define N_CELLS         16          /* brood cells: one per core */
#define MAX_ANTS        170
#define ANT_SPR         30          /* ant sprite size (px) */
#define N_ANG           64
#define N_FRM           4
#define ANT_SCALE       1.12
#define TUNNEL_W        17.0
#define STUB_W          12.0
#define FOOD_SEEDS      520
#define HUD_H           118
#define STEP            2.0         /* edge resample spacing */
#define MAX_PTS         200
#define TEMP_COOL       45.0        /* soil tint range, C */
#define TEMP_HOT        92.0
#define TEMP_FRANTIC    86.0        /* ants scurry above this (off again 4 C lower) */
#define NET_FULL        250e6       /* bytes/s for a full surface stream */
#define DISK_FULL       4e9         /* bytes/s for a full hauling line */

#define SWAP_FULL       50000.0     /* pages/s swapped for a full line to the cellar */
#define FORK_FULL       5000.0      /* new processes/s for the fastest egg laying */
#define PSI_FULL        30.0        /* % stalled for solid traffic jams */
#define N_BURROWS       14          /* surface burrows: one per TCP_PER_BURROW open connections */
#define TCP_PER_BURROW  8
#define BASE_Y          438.0       /* top of the wooden stand */

enum { S_L, S_ML, ENT, S_MR, S_R, J0, FOOD, JL1, JR1, NUR0, NUR1, JC, QUEEN, JL2, JR2, NV0, NV1, NV2, CELLAR, N_MAIN };
#define CELL0       N_MAIN                  /* brood cell centres */
#define ANCHOR0     (N_MAIN + N_CELLS)      /* where each cell's passage leaves its tunnel */
#define N_NODES     (N_MAIN + 2 * N_CELLS)
#define N_STORES    4                       /* the three drives and the swap cellar */
static const int STORE_NODE[N_STORES] = { NV0, NV1, NV2, CELLAR };

typedef struct { double x, y, rx, ry, oy; } node;   /* oy: chamber centre below the path node */
typedef struct {
    int a, b, surf, restricted, stub;
    double bend, w, len;
    int n;
    float px[MAX_PTS], py[MAX_PTS];
} edge;

enum { R_WORKER, R_HAULER, R_NET_IN, R_NET_OUT };
enum { IT_NONE, IT_LEAF, IT_WRITE, IT_READ, IT_PEBBLE, IT_SEED, N_ITEMS };

typedef struct {
    int alive, role, e, dir, carry, disk, route[32], nroute, ri, turning, dying, reverse_after;
    double s, speed, lane, x, y, ang, phase, alpha, pause, seed, v;
} ant;

typedef struct {
    double cpu, cell[N_CELLS], temp, heat, frantic, ram_used, ram_cache;
    double rd[SYS_NVME], wr[SYS_NVME], nvtemp[SYS_NVME], rx, tx, gact[2], gpow[2], gall;
    double swin, swout, forks, tasks, psi_cpu, psi_io, tcp;
} drive_t;

static node   nodes[N_NODES];
static edge   edges[64];
static int    n_edges;
static int    node_edges[N_NODES][8], node_deg[N_NODES];
static ant    ants[MAX_ANTS];
static double cell_ang[N_CELLS];
static int    frantic_on;
static double ram_shown_used = -1, ram_shown_cache = -1;

static cairo_surface_t *bg_cool, *bg_hot, *bg_mix, *fg_strip, *glass, *hud_cache, *food_layer;
static cairo_surface_t *ant_spr[2][N_ANG][N_FRM], *queen_spr, *larva_spr, *egg_spr, *item_spr[N_ITEMS];
static cairo_surface_t *glow_cell, *glow_nur[2], *glow_queen, *glow_nv, *glow_food, *glow_cellar, *plaque_cache;
static cairo_surface_t *burrow_spr[N_BURROWS][2];
static double burrow_x[N_BURROWS], burrow_a[N_BURROWS];
static int    burrows_shown;
static char   plaque_key[32];
static int    bg_mix_step = -1;
static char   hud_key[128];

typedef struct { float x, y, rx, ry, a, shade; } seed_t;
static seed_t seeds[FOOD_SEEDS];

typedef struct { double x, y, age, life, vx; } egg_t;
static egg_t  eggs[24];

static double frand(void) { return rand() / (double)RAND_MAX; }

static double wrap_pi(double a)
{
    while (a > M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
}

/* ground line (top of the soil) with the entrance mound around the shaft */
static double ground_y(double x)
{
    double m = (x - CX) / 30.0;
    return 131 + 3 * sin(x * 0.021 + 1) + 2 * sin(x * 0.057 + 0.4) - 12 * exp(-m * m);
}

/* ------------------------------------------------ layout */

static void add_node(int i, double x, double y, double rx, double ry)
{
    nodes[i] = (node){ x, y, rx, ry, 0 };
}

static void add_edge(int a, int b, double bend, double w, int surf, int restricted)
{
    edge *e = &edges[n_edges];
    double fx[400], fy[400], cum[400];
    int m = 399;
    e->a = a, e->b = b, e->bend = bend, e->w = w, e->surf = surf, e->restricted = restricted;
    e->stub = a >= CELL0 || b >= CELL0;
    const node *A = &nodes[a], *B = &nodes[b];
    double mx = (A->x + B->x) / 2, my = (A->y + B->y) / 2, dx = B->x - A->x, dy = B->y - A->y;
    double L = hypot(dx, dy), nx = -dy / L, ny = dx / L;
    double qx = mx + nx * bend, qy = my + ny * bend;
    for (int i = 0; i <= m; i++) {
        double t = i / (double)m, u = 1 - t;
        if (surf) {
            fx[i] = A->x + dx * t;
            fy[i] = ground_y(fx[i]) - 3.2;
            if (b == ENT || a == ENT) {       /* dip into the shaft mouth */
                double k = b == ENT ? t : u;
                fy[i] += 9 * pow(k, 6);
            }
        } else {
            double wig = (L > 40 ? 2.6 : 1.2) * sin(M_PI * t) * sin(M_PI * t * (L > 90 ? 3 : 2) + n_edges * 1.9);
            fx[i] = u * u * A->x + 2 * u * t * qx + t * t * B->x + nx * wig;
            fy[i] = u * u * A->y + 2 * u * t * qy + t * t * B->y + ny * wig;
        }
        cum[i] = i ? cum[i - 1] + hypot(fx[i] - fx[i - 1], fy[i] - fy[i - 1]) : 0;
    }
    e->len = cum[m];
    e->n = (int)(e->len / STEP) + 2;
    if (e->n > MAX_PTS)
        e->n = MAX_PTS;
    for (int k = 0, j = 0; k < e->n; k++) {
        double s = fmin(e->len, k * e->len / (e->n - 1));
        while (j < m - 1 && cum[j + 1] < s)
            j++;
        double f = (s - cum[j]) / fmax(1e-9, cum[j + 1] - cum[j]);
        e->px[k] = fx[j] + (fx[j + 1] - fx[j]) * f;
        e->py[k] = fy[j] + (fy[j + 1] - fy[j]) * f;
    }
    if (!e->stub) {                     /* brood-cell passages are scenery, not paths */
        node_edges[a][node_deg[a]++] = n_edges;
        node_edges[b][node_deg[b]++] = n_edges;
    }
    n_edges++;
}

/* position and unit tangent (a->b) at arc length s */
static void edge_at(const edge *e, double s, double *x, double *y, double *tx, double *ty)
{
    double f = clamp01(s / e->len) * (e->n - 1);
    int i = (int)f;
    if (i >= e->n - 1)
        i = e->n - 2;
    f -= i;
    *x = e->px[i] + (e->px[i + 1] - e->px[i]) * f;
    *y = e->py[i] + (e->py[i + 1] - e->py[i]) * f;
    int i0 = i > 1 ? i - 1 : 0, i1 = i + 2 < e->n ? i + 2 : e->n - 1;
    double dx = e->px[i1] - e->px[i0], dy = e->py[i1] - e->py[i0], l = hypot(dx, dy);
    *tx = l > 0 ? dx / l : 1;
    *ty = l > 0 ? dy / l : 0;
}

/*
 * Brood cells: alcoves off a tunnel wall, one per core (both SMT threads). Given as the
 * tunnel (a, b), how far along it (t) and which wall (side). Left half = CCD0, right = CCD1.
 */
static const struct { int a, b; double t, side; } CELLS[N_CELLS] = {
    { J0, JL1, 0.70, 1 }, { JL1, NUR0, 0.42, 1 }, { JL1, NUR0, 0.58, -1 }, { JC, NUR0, 0.45, 1 },
    { NUR0, JL2, 0.45, 1 }, { NUR0, JL2, 0.62, -1 }, { JL2, NV0, 0.5, 1 }, { JL2, NV1, 0.52, -1 },
    { J0, JR1, 0.70, -1 }, { JR1, NUR1, 0.42, -1 }, { JR1, NUR1, 0.58, 1 }, { JC, NUR1, 0.45, -1 },
    { NUR1, JR2, 0.45, -1 }, { NUR1, JR2, 0.62, 1 }, { JR2, NV2, 0.5, -1 }, { JR2, NV1, 0.52, 1 },
};

static int edge_between(int a, int b);

static void build_graph(void)
{
    add_node(S_L, -18, ground_y(-18) - 3, 0, 0);
    add_node(S_ML, 118, ground_y(118) - 3, 0, 0);
    add_node(ENT, CX, ground_y(CX) + 6, 0, 0);
    add_node(S_MR, 362, ground_y(362) - 3, 0, 0);
    add_node(S_R, 498, ground_y(498) - 3, 0, 0);
    add_node(J0, 240, 152, 0, 0);
    add_node(FOOD, 240, 197, 78, 28);
    add_node(JL1, 122, 168, 0, 0);
    add_node(JR1, 358, 168, 0, 0);
    add_node(NUR0, 100, 246, 44, 23);
    add_node(NUR1, 380, 246, 44, 23);
    add_node(JC, 240, 244, 0, 0);
    add_node(QUEEN, 240, 280, 64, 28);
    nodes[QUEEN].oy = 16;               /* her chamber hangs below the path in */
    add_node(JL2, 152, 334, 0, 0);
    add_node(JR2, 328, 334, 0, 0);
    add_node(NV0, 118, 382, 34, 18);
    add_node(NV1, 240, 374, 36, 18);
    add_node(NV2, 362, 382, 34, 18);
    add_node(CELLAR, 240, 420, 44, 13);

    add_edge(S_L, S_ML, 0, 0, 1, 1);
    add_edge(S_ML, ENT, 0, 0, 1, 1);
    add_edge(ENT, S_MR, 0, 0, 1, 1);
    add_edge(S_MR, S_R, 0, 0, 1, 1);
    add_edge(ENT, J0, 0, TUNNEL_W, 0, 1);
    add_edge(J0, FOOD, 3, TUNNEL_W, 0, 0);
    add_edge(J0, JL1, 10, TUNNEL_W, 0, 0);
    add_edge(J0, JR1, -10, TUNNEL_W, 0, 0);
    add_edge(JL1, NUR0, 14, TUNNEL_W, 0, 0);
    add_edge(JR1, NUR1, -14, TUNNEL_W, 0, 0);
    add_edge(FOOD, JC, -4, TUNNEL_W, 0, 0);
    add_edge(JC, QUEEN, 3, TUNNEL_W, 0, 0);
    add_edge(JC, NUR0, -6, TUNNEL_W * 0.85, 0, 0);
    add_edge(JC, NUR1, 6, TUNNEL_W * 0.85, 0, 0);
    add_edge(NUR0, JL2, -12, TUNNEL_W, 0, 0);
    add_edge(NUR1, JR2, 12, TUNNEL_W, 0, 0);
    add_edge(JL2, NV0, 8, TUNNEL_W, 0, 0);
    add_edge(JR2, NV2, -8, TUNNEL_W, 0, 0);
    add_edge(JL2, NV1, 12, TUNNEL_W, 0, 0);
    add_edge(JR2, NV1, -12, TUNNEL_W, 0, 0);
    add_edge(NV1, CELLAR, 0, TUNNEL_W * 0.8, 0, 0);

    for (int i = 0; i < N_CELLS; i++) {
        const edge *e = &edges[edge_between(CELLS[i].a, CELLS[i].b)];
        double x, y, tx, ty, off = TUNNEL_W / 2 + 10;
        edge_at(e, CELLS[i].t * e->len, &x, &y, &tx, &ty);
        if (e->a != CELLS[i].a)
            tx = -tx, ty = -ty;
        double nx = -ty * CELLS[i].side, ny = tx * CELLS[i].side;
        add_node(ANCHOR0 + i, x, y, 0, 0);
        add_node(CELL0 + i, x + nx * off, y + ny * off, 11, 8);
        add_edge(ANCHOR0 + i, CELL0 + i, 0, STUB_W, 0, 0);
        cell_ang[i] = atan2(ny, nx);
    }
}

/* shortest route (node list) between two nodes, surface allowed */
static int find_route(int from, int to, int *out, int cap)
{
    double dist[N_NODES];
    int prev[N_NODES], done[N_NODES] = { 0 };
    for (int i = 0; i < N_NODES; i++)
        dist[i] = 1e18, prev[i] = -1;
    dist[from] = 0;
    for (;;) {
        int u = -1;
        for (int i = 0; i < N_NODES; i++)
            if (!done[i] && dist[i] < 1e17 && (u < 0 || dist[i] < dist[u]))
                u = i;
        if (u < 0 || u == to)
            break;
        done[u] = 1;
        for (int k = 0; k < node_deg[u]; k++) {
            const edge *e = &edges[node_edges[u][k]];
            int v = e->a == u ? e->b : e->a;
            if (v >= N_MAIN && v != to)
                continue;                   /* don't route through brood cells */
            if (dist[u] + e->len < dist[v])
                dist[v] = dist[u] + e->len, prev[v] = u;
        }
    }
    int tmp[N_NODES], n = 0;
    for (int v = to; v >= 0 && n < N_NODES; v = prev[v])
        tmp[n++] = v;
    if (n > cap || tmp[n - 1] != from)
        return 0;
    for (int i = 0; i < n; i++)
        out[i] = tmp[n - 1 - i];
    return n;
}

static int edge_between(int a, int b)
{
    for (int k = 0; k < node_deg[a]; k++) {
        const edge *e = &edges[node_edges[a][k]];
        if ((e->a == a && e->b == b) || (e->b == a && e->a == b))
            return node_edges[a][k];
    }
    return -1;
}

/* ------------------------------------------------ sprites */

/* An ant pointing +x at the origin, body length ~17 units. frame: leg phase, pose 1 = antennae raised */
static void draw_ant(cairo_t *cr, int frame, int pose, double darkness)
{
    double ph = frame * M_PI / 2;
    double sw = sin(ph) * 0.38;
    rgb body = { 0.12 * darkness, 0.065 * darkness, 0.04 * darkness };
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);

    /* legs: tripod gait, group A = L1 R2 L3, group B = R1 L2 R3 */
    static const double base_x[3] = { 2.6, 1.5, 0.4 }, rest[3] = { 0.85, 1.6, 2.35 }, bend[3] = { -0.55, 0.15, 0.55 };
    cairo_set_line_width(cr, 0.95);
    set_rgb(cr, lerp(body, (rgb){ 0, 0, 0 }, 0.2));
    for (int side = -1; side <= 1; side += 2)
        for (int l = 0; l < 3; l++) {
            int group = (l + (side > 0)) & 1;
            double a = rest[l] + (group ? sw : -sw) * (l == 1 ? 0.7 : 1);
            double kx = base_x[l] + cos(a) * 4.3, ky = sin(a) * 4.3;
            double fx = kx + cos(a + bend[l]) * 4.6, fy = ky + sin(a + bend[l]) * 4.6;
            cairo_move_to(cr, base_x[l], side * 0.6);
            cairo_line_to(cr, kx, side * ky);
            cairo_line_to(cr, fx, side * fy);
        }
    cairo_stroke(cr);

    /* antennae: elbowed, waving in the raised pose */
    cairo_set_line_width(cr, 0.75);
    for (int side = -1; side <= 1; side += 2) {
        double spread = pose ? 0.75 : 0.35;
        double ex = 7.4 + cos(spread) * 3.0, ey = sin(spread) * 3.0 + 1.0;
        double tx = ex + cos(spread * 0.4 - (pose ? 0.2 : 0)) * 3.6, ty = ey - (pose ? 1.4 : 0.2) + sin(spread * 0.3) * 1.2;
        cairo_move_to(cr, 6.9, side * 0.9);
        cairo_line_to(cr, ex, side * ey);
        cairo_line_to(cr, tx, side * ty);
    }
    cairo_stroke(cr);

    /* gaster, petiole, thorax, head */
    struct { double x, rx, ry; } seg[4] = { { -5.0, 4.4, 3.2 }, { -0.9, 1.1, 0.9 }, { 1.6, 2.7, 1.55 }, { 5.6, 2.2, 1.95 } };
    for (int i = 0; i < 4; i++) {
        cairo_save(cr);
        cairo_translate(cr, seg[i].x, 0);
        cairo_scale(cr, seg[i].rx, seg[i].ry);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_pattern_t *g = cairo_pattern_create_radial(seg[i].x - seg[i].rx * 0.3, -seg[i].ry * 0.45, 0,
                                                          seg[i].x, 0, seg[i].rx * 1.1);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.55 * darkness + 0.08, 0.28 * darkness + 0.04, 0.14 * darkness);
        cairo_pattern_add_color_stop_rgb(g, 0.55, body.r * 1.4, body.g * 1.3, body.b * 1.2);
        cairo_pattern_add_color_stop_rgb(g, 1, body.r * 0.6, body.g * 0.6, body.b * 0.6);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    /* glossy glints on gaster and head */
    cairo_set_source_rgba(cr, 1, 0.9, 0.75, 0.55);
    cairo_arc(cr, -6.2, -1.3, 0.8, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_set_source_rgba(cr, 1, 0.9, 0.75, 0.4);
    cairo_arc(cr, 5.2, -0.9, 0.5, 0, 2 * M_PI);
    cairo_fill(cr);
    /* mandibles */
    cairo_set_line_width(cr, 0.7);
    set_rgb(cr, body);
    for (int side = -1; side <= 1; side += 2) {
        cairo_move_to(cr, 7.4, side * 0.9);
        cairo_curve_to(cr, 8.6, side * 1.1, 9.0, side * 0.5, 8.6, side * 0.1);
    }
    cairo_stroke(cr);
}

static cairo_surface_t *make_ant(double ang, int frame, int pose)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, ANT_SPR, ANT_SPR);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, ANT_SPR / 2.0, ANT_SPR / 2.0);
    cairo_rotate(cr, ang);
    cairo_scale(cr, ANT_SCALE, ANT_SCALE);
    /* a faint dark halo so ants read over bright tunnels, and a soft shadow */
    cairo_save(cr);
    cairo_translate(cr, 0.8, 1.2);
    cairo_scale(cr, 1, 0.55);
    cairo_pattern_t *g = cairo_pattern_create_radial(-1, 0, 0, -1, 0, 11);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.1, 0.04, 0.02, 0.35);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.1, 0.04, 0.02, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_restore(cr);
    draw_ant(cr, frame, pose, 1.0);
    cairo_destroy(cr);
    return s;
}

/* The queen: a big ant with a long banded gaster, facing left */
static cairo_surface_t *make_queen(void)
{
    const int w = 120, h = 70;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, w / 2.0 - 8, h / 2.0);
    cairo_rotate(cr, M_PI + 0.06);
    cairo_scale(cr, 2.7, 2.7);
    /* soft shadow on the chamber floor */
    cairo_save(cr);
    cairo_translate(cr, -6, -1.8);
    cairo_scale(cr, 17, 7);
    cairo_pattern_t *g = cairo_pattern_create_radial(0, 0, 0, 0, 0, 1);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.12, 0.04, 0.01, 0.45);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.12, 0.04, 0.01, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_restore(cr);
    draw_ant(cr, 1, 0, 1.7);
    /* long banded gaster over the ant's own */
    cairo_save(cr);
    cairo_translate(cr, -9.6, 0);
    cairo_scale(cr, 8.6, 4.9);
    cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
    cairo_restore(cr);
    g = cairo_pattern_create_radial(-11.5, 2.4, 0, -9.6, 0, 9.4);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.95, 0.62, 0.32);
    cairo_pattern_add_color_stop_rgb(g, 0.45, 0.52, 0.24, 0.10);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.16, 0.06, 0.03);
    cairo_set_source(cr, g);
    cairo_fill_preserve(cr);
    cairo_pattern_destroy(g);
    cairo_save(cr);
    cairo_clip(cr);
    cairo_set_line_width(cr, 0.7);
    for (int i = 0; i < 5; i++) {
        double x = -3.4 - i * 2.5;
        cairo_move_to(cr, x, -5.2);
        cairo_curve_to(cr, x - 1.3, -2, x - 1.3, 2, x, 5.2);
    }
    cairo_set_source_rgba(cr, 0.12, 0.04, 0.02, 0.55);
    cairo_stroke(cr);
    cairo_restore(cr);
    cairo_set_source_rgba(cr, 1, 0.94, 0.82, 0.75);
    cairo_save(cr);
    cairo_translate(cr, -12, 2.4);
    cairo_scale(cr, 2.0, 0.9);
    cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_fill(cr);
    cairo_destroy(cr);
    return s;
}

/* A plump curled larva */
static cairo_surface_t *make_larva(void)
{
    const int w = 18, h = 18;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, w / 2.0, h / 2.0);
    for (int i = 0; i < 7; i++) {
        double a = -2.4 + i * 0.62, r = 3.6, rr = 2.3 + 0.5 * sin(i / 6.0 * M_PI);
        double x = cos(a) * r, y = sin(a) * r;
        cairo_pattern_t *g = cairo_pattern_create_radial(x - 0.6, y - 0.7, 0, x, y, rr);
        cairo_pattern_add_color_stop_rgb(g, 0, 1, 0.98, 0.9);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.78, 0.7, 0.56);
        cairo_set_source(cr, g);
        cairo_arc(cr, x, y, rr, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    cairo_set_source_rgb(cr, 0.55, 0.36, 0.2);
    cairo_arc(cr, cos(-2.4) * 3.6 + 0.4, sin(-2.4) * 3.6 + 0.2, 0.8, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_destroy(cr);
    return s;
}

static cairo_surface_t *make_blob(int w, int h, double rx, double ry, rgb hi, rgb lo)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, w / 2.0, h / 2.0);
    cairo_save(cr);
    cairo_scale(cr, rx, ry);
    cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_pattern_t *g = cairo_pattern_create_radial(-rx * 0.35, -ry * 0.4, 0, 0, 0, fmax(rx, ry) * 1.1);
    cairo_pattern_add_color_stop_rgb(g, 0, hi.r, hi.g, hi.b);
    cairo_pattern_add_color_stop_rgb(g, 1, lo.r, lo.g, lo.b);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

static cairo_surface_t *make_leaf(void)
{
    const int w = 14, h = 14;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, w / 2.0, h / 2.0);
    cairo_rotate(cr, -0.6);
    cairo_move_to(cr, -5.5, 0);
    cairo_curve_to(cr, -2, -4.2, 3, -3.6, 5.5, 0);
    cairo_curve_to(cr, 3, 3.6, -2, 4.2, -5.5, 0);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, -4, 0, 4);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.62, 0.84, 0.30);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.22, 0.48, 0.14);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_move_to(cr, -5, 0);
    cairo_line_to(cr, 5, 0);
    cairo_set_line_width(cr, 0.6);
    cairo_set_source_rgba(cr, 0.85, 0.95, 0.6, 0.6);
    cairo_stroke(cr);
    cairo_destroy(cr);
    return s;
}

static cairo_surface_t *make_glow(double r, rgb c, double a0)
{
    int w = (int)ceil(r * 2) + 2;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, w);
    cairo_t *cr = cairo_create(s);
    cairo_pattern_t *g = cairo_pattern_create_radial(w / 2.0, w / 2.0, 0, w / 2.0, w / 2.0, r);
    cairo_pattern_add_color_stop_rgba(g, 0.0, c.r, c.g, c.b, a0);
    cairo_pattern_add_color_stop_rgba(g, 0.25, c.r, c.g, c.b, a0 * 0.6);
    cairo_pattern_add_color_stop_rgba(g, 0.6, c.r, c.g, c.b, a0 * 0.18);
    cairo_pattern_add_color_stop_rgba(g, 1.0, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

/* ------------------------------------------------ background */

static const rgb SOIL_COOL[6] = {
    { 0.22, 0.13, 0.08 }, { 0.33, 0.19, 0.11 }, { 0.42, 0.23, 0.12 },
    { 0.47, 0.31, 0.17 }, { 0.31, 0.18, 0.10 }, { 0.19, 0.11, 0.07 },
};

static rgb soil_col(int i, double heat)
{
    rgb c = SOIL_COOL[i];
    rgb h = { fmin(1, c.r * 1.35 + 0.06), c.g * 0.78, c.b * 0.55 };
    return lerp(c, h, heat);
}

static double band_y(int i, double x)
{
    static const double base[5] = { 168, 236, 300, 362, 426 };
    return base[i] + 7 * sin(x * 0.017 + i * 1.7) + 4 * sin(x * 0.043 + i * 2.9) + 2 * sin(x * 0.11 + i);
}

static void ellipse_path(cairo_t *cr, double x, double y, double rx, double ry)
{
    /* a chamber: slightly flattened floor */
    cairo_new_sub_path(cr);
    cairo_move_to(cr, x - rx, y + ry * 0.1);
    cairo_curve_to(cr, x - rx, y - ry * 0.75, x - rx * 0.55, y - ry, x, y - ry);
    cairo_curve_to(cr, x + rx * 0.55, y - ry, x + rx, y - ry * 0.75, x + rx, y + ry * 0.1);
    cairo_curve_to(cr, x + rx, y + ry * 0.7, x + rx * 0.6, y + ry * 0.95, x, y + ry * 0.95);
    cairo_curve_to(cr, x - rx * 0.6, y + ry * 0.95, x - rx, y + ry * 0.7, x - rx, y + ry * 0.1);
    cairo_close_path(cr);
}

/* carve every tunnel and chamber with one layer of stroke/fill (grow = extra width) */
static void carve(cairo_t *cr, double grow, double shrink_k)
{
    double dy = (1 - shrink_k) * 2.2;              /* inner light sits low: floor lit, roof in shadow */
    for (int i = 0; i < n_edges; i++) {
        const edge *e = &edges[i];
        if (e->surf)
            continue;
        for (int k = 0; k < e->n; k++) {
            double s = k * STEP;
            double wf = 1 + 0.13 * sin(s * 0.19 + i * 2.1) + 0.08 * sin(s * 0.47 + i * 0.7);
            if (k < 4 || k > e->n - 5)
                wf = 1;
            double r = fmax(0.5, (e->w * shrink_k * wf + grow) / 2);
            cairo_new_sub_path(cr);
            cairo_arc(cr, e->px[k], e->py[k] + dy, r, 0, 2 * M_PI);
        }
        cairo_fill(cr);
    }
    for (int i = J0; i < N_NODES; i++) {
        const node *n = &nodes[i];
        if (n->rx <= 0)
            continue;
        double k = shrink_k, rx = n->rx * k + grow / 2, ry = n->ry * k + grow / 2;
        if (rx < 1 || ry < 1)
            continue;
        ellipse_path(cr, n->x, n->y + n->oy + (1 - k) * n->ry * 0.3, rx, ry);
        cairo_fill(cr);
    }
}

#define TUNNEL_LAYERS   11

static rgb tunnel_ramp(double f, rgb dark, rgb mid, rgb core, rgb hot)
{
    return f < 0.45 ? lerp(dark, mid, f / 0.45) : f < 0.82 ? lerp(mid, core, (f - 0.45) / 0.37) : lerp(core, hot, (f - 0.82) / 0.18);
}

static void tunnel_palette(double heat, rgb *dark, rgb *mid, rgb *core, rgb *hot)
{
    *dark = lerp((rgb){ 0.20, 0.10, 0.05 }, (rgb){ 0.28, 0.08, 0.03 }, heat);
    *mid  = lerp((rgb){ 0.55, 0.32, 0.15 }, (rgb){ 0.66, 0.26, 0.10 }, heat);
    *core = lerp((rgb){ 0.86, 0.60, 0.33 }, (rgb){ 0.98, 0.52, 0.24 }, heat);
    *hot  = lerp((rgb){ 0.98, 0.82, 0.55 }, (rgb){ 1.00, 0.72, 0.42 }, heat);
}

/* a short burrow dug from the surface: one per few open TCP connections */
static cairo_surface_t *make_burrow(int i, double heat)
{
    const int w = 40, h = 34, mx = 20, my = 6;
    double dir = burrow_x[i] < CX ? -1 : 1, depth = 12 + (i * 7 % 7);
    double ex = dir * (5 + (i % 3)), ey = depth;
    rgb dark, mid, core, hot;
    tunnel_palette(heat, &dark, &mid, &core, &hot);
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, mx, my);
    /* spoil heap of dug-out earth beside the mouth */
    rgb soil = soil_col(0, heat);
    srand(100 + i);
    for (int k = 0; k < 9; k++) {
        double x = -dir * (4 + frand() * 6), y = -0.5 - frand() * 2.2 * (1 - fabs(x) / 12), r = 1.1 + frand() * 1.3;
        rgb c = lerp(soil, (rgb){ 0.7, 0.45, 0.25 }, 0.25 + 0.3 * frand());
        cairo_arc(cr, x, y, r, 0, 2 * M_PI);
        set_rgb(cr, c);
        cairo_fill(cr);
    }
    cairo_save(cr);
    cairo_rectangle(cr, -mx, -0.5, w, h);
    cairo_clip(cr);
    for (int l = -2; l < 7; l++) {
        double k = l < 0 ? 1 : 1 - 0.8 * (l + 1) / 7.0, grow = l == -2 ? 4.5 : l == -1 ? 1 : 0;
        rgb c = l == -2 ? (rgb){ 0.04, 0.015, 0 } : l == -1 ? dark : tunnel_ramp((l + 1) / 7.0 * 0.85, dark, mid, core, hot);
        double r = (7.5 * k + grow) / 2, dy = (1 - k) * 1.5;
        for (int j = 0; j <= 10; j++) {
            double f = j / 10.0;
            cairo_new_sub_path(cr);
            cairo_arc(cr, ex * f, -3 + (ey + 3) * f + dy, r, 0, 2 * M_PI);
        }
        cairo_new_sub_path(cr);
        cairo_arc(cr, ex + dir * 1.5, ey + dy, r * 1.35, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, c.r * 0.85, c.g * 0.85, c.b * 0.85, l == -2 ? 0.5 : 1);
        cairo_fill(cr);
    }
    cairo_restore(cr);
    cairo_destroy(cr);
    return s;
}

static void soil_region(cairo_t *cr)
{
    cairo_new_path(cr);
    cairo_move_to(cr, 0, ground_y(0));
    for (int x = 4; x <= SIZE; x += 4)
        cairo_line_to(cr, x, ground_y(x));
    cairo_line_to(cr, SIZE, SIZE);
    cairo_line_to(cr, 0, SIZE);
    cairo_close_path(cr);
}

static cairo_surface_t *build_bg(double heat)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(s);
    srand(11);

    /* sky: the air above the soil, dusk seen through the glass */
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, 0, 140);
    cairo_pattern_add_color_stop_rgb(g, 0.00, 0.06, 0.04, 0.09);
    cairo_pattern_add_color_stop_rgb(g, 0.45, 0.14, 0.07, 0.11);
    cairo_pattern_add_color_stop_rgb(g, 0.80, 0.42 + 0.1 * heat, 0.21, 0.13);
    cairo_pattern_add_color_stop_rgb(g, 1.00, 0.72 + 0.1 * heat, 0.42, 0.22);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    /* a low warm sun-glow behind the mound */
    g = cairo_pattern_create_radial(CX, 128, 0, CX, 128, 200);
    cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.72, 0.42, 0.35);
    cairo_pattern_add_color_stop_rgba(g, 1, 1, 0.72, 0.42, 0);
    cairo_set_source(cr, g);
    cairo_rectangle(cr, 0, 0, SIZE, 140);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    /* soil strata */
    cairo_save(cr);
    soil_region(cr);
    cairo_clip(cr);
    set_rgb(cr, soil_col(0, heat));
    cairo_paint(cr);
    for (int i = 0; i < 5; i++) {
        cairo_new_path(cr);
        cairo_move_to(cr, 0, band_y(i, 0));
        for (int x = 4; x <= SIZE; x += 4)
            cairo_line_to(cr, x, band_y(i, x));
        cairo_line_to(cr, SIZE, SIZE);
        cairo_line_to(cr, 0, SIZE);
        cairo_close_path(cr);
        set_rgb(cr, soil_col(i + 1, heat));
        cairo_fill(cr);
        /* a soft darker seam along each boundary */
        cairo_new_path(cr);
        cairo_move_to(cr, 0, band_y(i, 0));
        for (int x = 4; x <= SIZE; x += 4)
            cairo_line_to(cr, x, band_y(i, x));
        cairo_set_line_width(cr, 3);
        cairo_set_source_rgba(cr, 0.05, 0.02, 0.01, 0.25);
        cairo_stroke(cr);
    }
    /* depth shading: darker toward the bottom */
    g = cairo_pattern_create_linear(0, 130, 0, SIZE);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.02, 0.0, 0.0, 0.35);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);

    /* grains */
    for (int i = 0; i < 16000; i++) {
        double x = frand() * SIZE, y = 120 + frand() * 360, v = frand();
        double a = 0.08 + 0.18 * frand();
        if (v < 0.5)
            cairo_set_source_rgba(cr, 0.03, 0.01, 0.0, a);
        else
            cairo_set_source_rgba(cr, 0.95, 0.75 - 0.15 * heat, 0.5 - 0.2 * heat, a * 0.7);
        cairo_rectangle(cr, x, y, 1 + (frand() < 0.3), 1);
        cairo_fill(cr);
    }
    /* pebbles */
    for (int i = 0; i < 150; i++) {
        double x = frand() * SIZE, y = 140 + frand() * 340;
        double r = 1.5 + pow(frand(), 3) * 7, sq = 0.6 + frand() * 0.35, rot = frand() * M_PI;
        double tone = 0.3 + frand() * 0.4;
        rgb pc = lerp((rgb){ tone, tone * 0.85, tone * 0.72 }, soil_col(3, heat), 0.35);
        cairo_save(cr);
        cairo_translate(cr, x + 1, y + 1.4);
        cairo_rotate(cr, rot);
        cairo_scale(cr, r, r * sq);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, 0.03, 0.01, 0, 0.45);
        cairo_fill(cr);
        cairo_save(cr);
        cairo_translate(cr, x, y);
        cairo_rotate(cr, rot);
        cairo_scale(cr, r, r * sq);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        g = cairo_pattern_create_radial(x - r * 0.35, y - r * 0.4, 0, x, y, r * 1.1);
        cairo_pattern_add_color_stop_rgb(g, 0, fmin(1, pc.r * 1.6), fmin(1, pc.g * 1.6), fmin(1, pc.b * 1.6));
        cairo_pattern_add_color_stop_rgb(g, 1, pc.r * 0.7, pc.g * 0.7, pc.b * 0.7);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    /* roots hanging from the grass */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < 26; i++) {
        double x = 20 + frand() * 440, y = ground_y(x) + 1, len = 16 + frand() * 38, a = M_PI / 2 + (frand() - 0.5) * 0.7;
        if (fabs(x - CX) < 36)
            continue;
        cairo_move_to(cr, x, y);
        for (int k = 0; k < 6; k++) {
            a += (frand() - 0.5) * 0.6;
            x += cos(a) * len / 6;
            y += sin(a) * len / 6;
            cairo_line_to(cr, x, y);
        }
        cairo_set_line_width(cr, 0.8 + frand() * 1.1);
        cairo_set_source_rgba(cr, 0.10, 0.06, 0.03, 0.75);
        cairo_stroke(cr);
    }

    /* tunnels and chambers, backlit: light shining through the gaps in the soil */
    rgb dark, mid, core, hot;
    tunnel_palette(heat, &dark, &mid, &core, &hot);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_source_rgba(cr, 0.04, 0.015, 0.0, 0.55);            /* packed walls */
    carve(cr, 6, 1);
    set_rgb(cr, dark);
    carve(cr, 1.5, 1);
    for (int i = 0; i < TUNNEL_LAYERS; i++) {
        double f = (i + 1.0) / TUNNEL_LAYERS;
        rgb c = tunnel_ramp(f * 0.93, dark, mid, core, hot);
        set_rgb(cr, c);
        carve(cr, 0, 1 - 0.7 * f);
    }
    /* the shaft is open to the sky at the top */
    cairo_restore(cr);

    /* floor grit in tunnels */
    for (int i = 0; i < n_edges; i++) {
        const edge *e = &edges[i];
        if (e->surf)
            continue;
        for (int k = 2; k < e->n - 2; k += 3) {
            if (frand() < 0.5)
                continue;
            double ox = (frand() - 0.5) * e->w * 0.6, oy = (frand() - 0.5) * e->w * 0.6;
            cairo_set_source_rgba(cr, 0.25, 0.12, 0.05, 0.25 + frand() * 0.2);
            cairo_arc(cr, e->px[k] + ox, e->py[k] + oy, 0.6 + frand() * 0.8, 0, 2 * M_PI);
            cairo_fill(cr);
        }
    }

    /* chamber furnishings: fungus beds in the nurseries, crumb stashes by the drives */
    for (int k = 0; k < 2; k++) {
        const node *n = &nodes[NUR0 + k];
        rgb fc = k ? (rgb){ 0.92, 0.78, 0.56 } : (rgb){ 0.72, 0.84, 0.86 };
        for (int i = 0; i < 70; i++) {
            double u = frand() * 2 - 1, x = n->x + u * n->rx * 0.8;
            double top = n->y + n->ry * (0.15 + 0.35 * u * u), y = top + frand() * (n->y + n->ry * 0.85 - top);
            double r = 1.8 + frand() * 3.2;
            g = cairo_pattern_create_radial(x - r * 0.3, y - r * 0.4, 0, x, y, r);
            cairo_pattern_add_color_stop_rgba(g, 0, fc.r, fc.g, fc.b, 0.95);
            cairo_pattern_add_color_stop_rgba(g, 1, fc.r * 0.45, fc.g * 0.4, fc.b * 0.4, 0.9);
            cairo_set_source(cr, g);
            cairo_arc(cr, x, y, r, 0, 2 * M_PI);
            cairo_fill(cr);
            cairo_pattern_destroy(g);
        }
    }
    for (int k = 0; k < 3; k++) {
        const node *n = &nodes[NV0 + k];
        for (int i = 0; i < 46; i++) {
            double u = frand() * 2 - 1, x = n->x + u * n->rx * 0.75;
            double y = n->y + n->ry * (0.75 - 0.45 * (1 - u * u) * frand());
            double r = 1.4 + frand() * 1.8, v = frand();
            rgb c = v < 0.5 ? (rgb){ 0.95, 0.68, 0.30 } : (rgb){ 0.78, 0.86, 0.92 };
            cairo_set_source_rgba(cr, c.r * 0.35, c.g * 0.3, c.b * 0.3, 0.8);
            cairo_arc(cr, x + 0.6, y + 0.8, r, 0, 2 * M_PI);
            cairo_fill(cr);
            set_rgb(cr, lerp(c, (rgb){ 0.4, 0.25, 0.15 }, 0.25 + 0.3 * frand()));
            cairo_arc(cr, x, y, r, 0, 2 * M_PI);
            cairo_fill(cr);
        }
    }

    /* the swap cellar: a deep stash of seeds under hanging roots */
    {
        const node *n = &nodes[CELLAR];
        for (int i = 0; i < 70; i++) {
            double u = frand() * 2 - 1, x = n->x + u * n->rx * 0.8;
            double y = n->y + n->ry * (0.7 - 0.6 * (1 - u * u) * frand());
            double r = 1.6 + frand() * 1.2;
            cairo_save(cr);
            cairo_translate(cr, x, y);
            cairo_rotate(cr, frand() * M_PI);
            cairo_scale(cr, r * 1.4, r);
            cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
            cairo_restore(cr);
            set_rgb(cr, lerp((rgb){ 0.95, 0.72, 0.36 }, (rgb){ 0.45, 0.26, 0.12 }, 0.2 + 0.5 * frand()));
            cairo_fill(cr);
        }
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        for (int i = 0; i < 7; i++) {
            double x = n->x - 34 + i * 11 + frand() * 4, y = n->y - n->ry - 4, len = 6 + frand() * 8;
            if (fabs(x - n->x) < 9)
                continue;
            cairo_move_to(cr, x, y);
            cairo_curve_to(cr, x + 1.5, y + len * 0.4, x - 1.5, y + len * 0.7, x + (frand() - 0.5) * 3, y + len);
            cairo_set_line_width(cr, 1.0 + frand() * 0.8);
            cairo_set_source_rgba(cr, 0.16, 0.09, 0.04, 0.85);
            cairo_stroke(cr);
        }
    }

    /* grass and the entrance mound's crest */
    for (int i = 0; i < 240; i++) {
        double x = frand() * SIZE, y = ground_y(x) + 1.5;
        if (fabs(x - CX) < 14)
            continue;
        double h = 5 + pow(frand(), 2) * 22 * (fabs(x - CX) < 50 ? 0.4 : 1), lean = (frand() - 0.5) * 9;
        double sh = 0.6 + frand() * 0.4;
        cairo_move_to(cr, x - 1.2, y);
        cairo_curve_to(cr, x - 0.6, y - h * 0.5, x + lean * 0.5, y - h * 0.8, x + lean, y - h);
        cairo_curve_to(cr, x + lean * 0.4 + 0.4, y - h * 0.75, x + 0.6, y - h * 0.4, x + 1.2, y);
        cairo_close_path(cr);
        cairo_set_source_rgb(cr, 0.10 * sh + 0.03, 0.08 * sh + 0.02, 0.05 * sh);
        cairo_fill(cr);
    }
    cairo_destroy(cr);
    return s;
}

/* foreground grass tufts (in front of the surface ants) */
static cairo_surface_t *build_fg(void)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, 60);
    cairo_t *cr = cairo_create(s);
    srand(23);
    cairo_translate(cr, 0, -100);
    for (int i = 0; i < 38; i++) {
        double x = frand() * SIZE, y = ground_y(x) + 4;
        if (fabs(x - CX) < 40)
            continue;
        double h = 6 + frand() * 12, lean = (frand() - 0.5) * 10;
        cairo_move_to(cr, x - 1.1, y);
        cairo_curve_to(cr, x - 0.5, y - h * 0.5, x + lean * 0.5, y - h * 0.8, x + lean, y - h);
        cairo_curve_to(cr, x + lean * 0.4 + 0.3, y - h * 0.75, x + 0.5, y - h * 0.4, x + 1.1, y);
        cairo_close_path(cr);
        cairo_set_source_rgb(cr, 0.07, 0.05, 0.035);
        cairo_fill(cr);
    }
    cairo_destroy(cr);
    return s;
}

/* the glass: a wooden frame ring, inner shadow and reflections */
static cairo_surface_t *build_glass(void)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cairo_t *cr = cairo_create(s);
    /* inner shadow */
    cairo_pattern_t *g = cairo_pattern_create_radial(CX, CY, R_GLASS - 34, CX, CY, R_GLASS);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.03, 0.01, 0.0, 0.6);
    cairo_set_source(cr, g);
    cairo_arc(cr, CX, CY, R_GLASS, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    /* frame */
    cairo_set_fill_rule(cr, CAIRO_FILL_RULE_EVEN_ODD);
    cairo_rectangle(cr, 0, 0, SIZE, SIZE);
    cairo_arc(cr, CX, CY, R_GLASS, 0, 2 * M_PI);
    g = cairo_pattern_create_radial(CX, CY, R_GLASS, CX, CY, 242);
    cairo_pattern_add_color_stop_rgb(g, 0.0, 0.09, 0.05, 0.03);
    cairo_pattern_add_color_stop_rgb(g, 0.25, 0.36, 0.22, 0.12);
    cairo_pattern_add_color_stop_rgb(g, 0.55, 0.24, 0.14, 0.08);
    cairo_pattern_add_color_stop_rgb(g, 1.0, 0.10, 0.06, 0.04);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    /* the wooden stand the farm sits in, with a brass name plaque */
    cairo_set_fill_rule(cr, CAIRO_FILL_RULE_WINDING);
    cairo_save(cr);
    cairo_arc(cr, CX, CY, 241, 0, 2 * M_PI);
    cairo_clip(cr);
    g = cairo_pattern_create_linear(0, BASE_Y - 10, 0, BASE_Y);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.03, 0.01, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.03, 0.01, 0, 0.55);
    cairo_set_source(cr, g);
    cairo_rectangle(cr, 0, BASE_Y - 10, SIZE, 10);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    g = cairo_pattern_create_linear(0, BASE_Y, 0, SIZE);
    cairo_pattern_add_color_stop_rgb(g, 0.0, 0.40, 0.25, 0.14);
    cairo_pattern_add_color_stop_rgb(g, 0.08, 0.30, 0.18, 0.10);
    cairo_pattern_add_color_stop_rgb(g, 1.0, 0.13, 0.07, 0.04);
    cairo_set_source(cr, g);
    cairo_rectangle(cr, 0, BASE_Y, SIZE, SIZE - BASE_Y);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    srand(41);
    for (int i = 0; i < 16; i++) {                  /* wood grain */
        double y = BASE_Y + 3 + i * 2.6 + frand() * 1.5;
        cairo_move_to(cr, 0, y);
        for (int x = 0; x <= SIZE; x += 24)
            cairo_line_to(cr, x, y + 1.2 * sin(x * 0.03 + i * 1.7));
        cairo_set_line_width(cr, 0.6 + frand() * 0.6);
        cairo_set_source_rgba(cr, 0.06, 0.03, 0.01, 0.25 + frand() * 0.2);
        cairo_stroke(cr);
    }
    cairo_move_to(cr, 0, BASE_Y + 0.5);
    cairo_line_to(cr, SIZE, BASE_Y + 0.5);
    cairo_set_line_width(cr, 1);
    cairo_set_source_rgba(cr, 1, 0.8, 0.55, 0.3);
    cairo_stroke(cr);
    cairo_restore(cr);
    /* plaque */
    {
        double pw = 150, ph = 28, px = CX - pw / 2, py = BASE_Y + 6, r = 5;
        cairo_new_sub_path(cr);
        cairo_arc(cr, px + pw - r, py + r, r, -M_PI / 2, 0);
        cairo_arc(cr, px + pw - r, py + ph - r, r, 0, M_PI / 2);
        cairo_arc(cr, px + r, py + ph - r, r, M_PI / 2, M_PI);
        cairo_arc(cr, px + r, py + r, r, M_PI, 1.5 * M_PI);
        cairo_close_path(cr);
        g = cairo_pattern_create_linear(0, py, 0, py + ph);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.93, 0.78, 0.45);
        cairo_pattern_add_color_stop_rgb(g, 0.5, 0.74, 0.55, 0.26);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.55, 0.38, 0.16);
        cairo_set_source(cr, g);
        cairo_fill_preserve(cr);
        cairo_pattern_destroy(g);
        cairo_set_line_width(cr, 1.2);
        cairo_set_source_rgba(cr, 0.25, 0.14, 0.05, 0.8);
        cairo_stroke(cr);
        for (int k = -1; k <= 1; k += 2) {           /* screws */
            cairo_arc(cr, CX + k * (pw / 2 - 7), py + ph / 2, 1.8, 0, 2 * M_PI);
            cairo_set_source_rgba(cr, 0.35, 0.22, 0.08, 0.9);
            cairo_fill(cr);
        }
    }
    srand((unsigned)time(NULL));

    /* frame highlight, upper left */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_arc(cr, CX, CY, R_GLASS + 4, M_PI * 1.05, M_PI * 1.45);
    cairo_set_line_width(cr, 2.2);
    cairo_set_source_rgba(cr, 1, 0.8, 0.55, 0.22);
    cairo_stroke(cr);
    /* glass reflections */
    for (int i = 0; i < 6; i++) {
        cairo_arc(cr, CX, CY, R_GLASS - 16, M_PI * 1.08, M_PI * 1.36);
        cairo_set_line_width(cr, 14 - i * 2.2);
        cairo_set_source_rgba(cr, 1, 0.96, 0.9, 0.018 + i * 0.006);
        cairo_stroke(cr);
    }
    cairo_arc(cr, CX, CY, R_GLASS - 12, M_PI * 0.10, M_PI * 0.22);
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgba(cr, 1, 0.96, 0.9, 0.05);
    cairo_stroke(cr);
    cairo_destroy(cr);
    return s;
}

/* seeds of the food store, bottom first, so the first n make a pile */
static int by_height(const void *a, const void *b)
{
    const seed_t *p = a, *q = b;
    double ka = p->y - p->shade * 0, kb = q->y;
    return ka < kb ? 1 : ka > kb ? -1 : 0;
}

static void build_seeds(void)
{
    const node *f = &nodes[FOOD];
    srand(31);
    for (int i = 0; i < FOOD_SEEDS; i++) {
        double x, y;
        do {
            x = (frand() * 2 - 1) * (f->rx - 5);
            y = (frand() * 2 - 1) * (f->ry - 4);
        } while ((x * x) / ((f->rx - 5) * (f->rx - 5)) + (y * y) / ((f->ry - 4) * (f->ry - 4)) > 1);
        seeds[i].x = (float)(f->x + x);
        seeds[i].y = (float)(f->y + y + 1.5 + (frand() - 0.5) * 3.5);   /* bumpy pile surface */
        seeds[i].rx = (float)(2.2 + frand() * 1.5);
        seeds[i].ry = (float)(seeds[i].rx * (0.55 + frand() * 0.2));
        seeds[i].a = (float)(frand() * M_PI);
        seeds[i].shade = (float)frand();
    }
    qsort(seeds, FOOD_SEEDS, sizeof(seed_t), by_height);
    food_layer = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, (int)(f->rx * 2 + 8), (int)(f->ry * 2 + 8));
    srand((unsigned)time(NULL));
}

static void draw_food(int n_used, int n_cache)
{
    const node *f = &nodes[FOOD];
    cairo_t *cr = cairo_create(food_layer);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_translate(cr, -(f->x - f->rx - 4), -(f->y - f->ry - 4));
    ellipse_path(cr, f->x, f->y, f->rx - 2, f->ry - 1.5);
    cairo_clip(cr);
    for (int i = 0; i < n_used + n_cache && i < FOOD_SEEDS; i++) {
        const seed_t *sd = &seeds[i];
        int cache = i >= n_used;
        rgb hi = cache ? (rgb){ 0.95, 0.86, 0.68 } : lerp((rgb){ 1.0, 0.82, 0.42 }, (rgb){ 0.92, 0.58, 0.26 }, sd->shade);
        rgb lo = cache ? (rgb){ 0.58, 0.46, 0.34 } : lerp((rgb){ 0.56, 0.32, 0.12 }, (rgb){ 0.42, 0.20, 0.08 }, sd->shade);
        double rx = sd->rx * (cache ? 0.8 : 1), ry = sd->ry * (cache ? 0.95 : 1);
        cairo_save(cr);
        cairo_translate(cr, sd->x + 0.5, sd->y + 0.8);
        cairo_rotate(cr, sd->a);
        cairo_scale(cr, rx, ry);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, 0.08, 0.03, 0.01, 0.5);
        cairo_fill(cr);
        cairo_save(cr);
        cairo_translate(cr, sd->x, sd->y);
        cairo_rotate(cr, sd->a);
        cairo_scale(cr, rx, ry);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_pattern_t *g = cairo_pattern_create_radial(sd->x - rx * 0.3, sd->y - ry * 0.5, 0, sd->x, sd->y, rx * 1.1);
        cairo_pattern_add_color_stop_rgb(g, 0, hi.r, hi.g, hi.b);
        cairo_pattern_add_color_stop_rgb(g, 1, lo.r, lo.g, lo.b);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    cairo_destroy(cr);
}

static void build_caches(void)
{
    build_graph();
    for (int p = 0; p < 2; p++)
        for (int a = 0; a < N_ANG; a++)
            for (int f = 0; f < N_FRM; f++)
                ant_spr[p][a][f] = make_ant(a * 2 * M_PI / N_ANG, f, p);
    queen_spr = make_queen();
    larva_spr = make_larva();
    egg_spr = make_blob(8, 8, 2.4, 1.6, (rgb){ 1, 1, 0.97 }, (rgb){ 0.8, 0.78, 0.7 });
    item_spr[IT_LEAF] = make_leaf();
    item_spr[IT_WRITE] = make_blob(10, 10, 2.8, 2.2, (rgb){ 1.0, 0.86, 0.45 }, (rgb){ 0.72, 0.38, 0.12 });
    item_spr[IT_READ] = make_blob(10, 10, 2.6, 2.3, (rgb){ 0.92, 0.98, 1.0 }, (rgb){ 0.45, 0.62, 0.78 });
    item_spr[IT_PEBBLE] = make_blob(10, 10, 2.4, 2.0, (rgb){ 0.95, 0.88, 0.78 }, (rgb){ 0.5, 0.42, 0.34 });
    item_spr[IT_SEED] = make_blob(12, 12, 3.6, 2.3, (rgb){ 1.0, 0.84, 0.46 }, (rgb){ 0.62, 0.34, 0.10 });
    static const double bx[N_BURROWS] = { 184, 296, 128, 352, 100, 380, 70, 410, 206, 274, 56, 424, 142, 338 };
    for (int i = 0; i < N_BURROWS; i++) {
        burrow_x[i] = bx[i];
        burrow_spr[i][0] = make_burrow(i, 0);
        burrow_spr[i][1] = make_burrow(i, 1);
    }
    plaque_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 150, 28);
    glow_cell = make_glow(30, (rgb){ 1.0, 0.72, 0.32 }, 0.85);
    glow_nur[0] = make_glow(72, (rgb){ 0.30, 0.72, 1.0 }, 0.75);
    glow_nur[1] = make_glow(72, (rgb){ 1.0, 0.62, 0.22 }, 0.75);
    glow_queen = make_glow(88, (rgb){ 1.0, 0.62, 0.40 }, 0.7);
    glow_nv = make_glow(52, (rgb){ 1.0, 0.85, 0.6 }, 0.6);
    glow_cellar = make_glow(56, (rgb){ 1.0, 0.78, 0.36 }, 0.7);
    glow_food = make_glow(96, (rgb){ 1.0, 0.8, 0.45 }, 0.3);
    bg_cool = build_bg(0);
    bg_hot = build_bg(1);
    bg_mix = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    fg_strip = build_fg();
    glass = build_glass();
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, HUD_H);
    build_seeds();
    srand((unsigned)time(NULL));
}

/* ---------------------------------------------------------------- simulation */

static int count_role(int role, int disk)
{
    int n = 0;
    for (int i = 0; i < MAX_ANTS; i++)
        if (ants[i].alive && !ants[i].dying && ants[i].role == role && (role != R_HAULER || ants[i].disk == disk))
            n++;
    return n;
}

static ant *new_ant(int role)
{
    for (int i = 0; i < MAX_ANTS; i++)
        if (!ants[i].alive) {
            ant *a = &ants[i];
            memset(a, 0, sizeof(*a));
            a->alive = 1;
            a->role = role;
            a->speed = 0.82 + frand() * 0.36;
            a->lane = (frand() - 0.5) * 5;
            a->seed = frand() * 100;
            a->alpha = 0;
            a->x = -1e9;
            return a;
        }
    return NULL;
}

/* put an ant on the first edge of its route */
static void start_route(ant *a)
{
    a->ri = 0;
    int e = edge_between(a->route[0], a->route[1]);
    a->e = e;
    if (edges[e].a == a->route[0])
        a->dir = 1, a->s = 0;
    else
        a->dir = -1, a->s = edges[e].len;
}

/* choose where to go at a node. Returns next edge or -1 to stop here (pause/turn/finish). */
static int choose_next(ant *a, int node_i, int from_e, const drive_t *d)
{
    if (a->role != R_WORKER) {
        a->ri++;
        if (a->ri + 1 >= a->nroute)
            return -1;
        return edge_between(a->route[a->ri], a->route[a->ri + 1]);
    }
    int cand[8], nc = 0;
    double w[8], wsum = 0;
    for (int k = 0; k < node_deg[node_i]; k++) {
        int e = node_edges[node_i][k];
        if (e == from_e || edges[e].restricted)
            continue;
        double wt = nodes[node_i].rx > 20 ? 1 : 1 + 0.5 * d->cpu;
        cand[nc] = e;
        w[nc++] = wt;
        wsum += wt;
    }
    /* sometimes linger or double back in a chamber */
    if (nc == 0 || (nodes[node_i].rx > 20 && frand() < 0.12))
        return -1;
    double r = frand() * wsum;
    for (int i = 0; i < nc; i++) {
        r -= w[i];
        if (r <= 0)
            return cand[i];
    }
    return cand[nc - 1];
}

static double disk_level(double bps) { return clamp01(log10(1 + bps / 50e3) / log10(1 + DISK_FULL / 50e3)); }
static double net_level(double bps) { return clamp01(log10(1 + bps / 2e3) / log10(1 + NET_FULL / 2e3)); }
static double swap_level(double pps) { return pps < 2 ? 0 : clamp01(log10(1 + pps) / log10(1 + SWAP_FULL)); }

/* a store's traffic: up = reads / swap-in, down = writes / swap-out (0..1) */
static void store_levels(const drive_t *d, int k, double *up, double *down)
{
    if (k < SYS_NVME) {
        *up = disk_level(d->rd[k]);
        *down = disk_level(d->wr[k]);
    } else {
        *up = swap_level(d->swin);
        *down = swap_level(d->swout);
    }
}

static int store_target(const drive_t *d, int k)
{
    double up, down;
    store_levels(d, k, &up, &down);
    return up + down > 0.03 ? (int)ceil(fmin(5, (up + down) * 6)) : 0;
}

static int store_item(int k, int down) { return k == SYS_NVME ? IT_SEED : down ? IT_WRITE : IT_READ; }

static int worker_target(double cpu) { return 4 + (int)(50 * pow(cpu, 0.8)); }

static void arrive_end(ant *a, const drive_t *d)
{
    int at = a->route[a->nroute - 1];
    switch (a->role) {
    case R_NET_IN:                      /* delivered into the food store */
        a->dying = 1;
        a->carry = IT_NONE;
        a->pause = 0.6;
        break;
    case R_NET_OUT:                     /* walked off the glass */
        a->alive = 0;
        break;
    case R_HAULER: {
        int down = at == FOOD;          /* at the store: head back down */
        double up_l, dn_l;
        store_levels(d, a->disk, &up_l, &dn_l);
        double m = fmax(1e-6, fmax(up_l, dn_l));
        if (down && count_role(R_HAULER, a->disk) > store_target(d, a->disk)) {
            a->dying = 1;
            a->pause = 0.5;
            break;
        }
        a->carry = down ? (frand() < dn_l / m && dn_l > 0.02 ? store_item(a->disk, 1) : IT_NONE)
                        : (frand() < up_l / m && up_l > 0.02 ? store_item(a->disk, 0) : IT_NONE);
        int to = down ? STORE_NODE[a->disk] : FOOD;
        a->nroute = find_route(at, to, a->route, 32);
        a->pause = 0.45 + frand() * 0.4;  /* turns round on the spot while it loads up */
        start_route(a);
        break;
    }
    default:
        break;
    }
}

static void ant_step(ant *a, double dt, double v, const drive_t *d, double t)
{
    /* fades */
    if (a->dying) {
        a->alpha -= dt * 1.6;
        if (a->alpha <= 0) {
            a->alive = 0;
            return;
        }
    } else if (a->alpha < 1)
        a->alpha = fmin(1, a->alpha + dt * 1.6);

    if (a->pause > 0) {
        a->pause -= dt;
        if (a->pause <= 0 && a->reverse_after) {
            a->reverse_after = 0;
            if (a->role == R_WORKER) {
                a->dir = -a->dir;
            } else {
                start_route(a);
            }
        }
    } else {
        edge *E = &edges[a->e];
        double dist = v * dt;
        a->s += a->dir * dist;
        a->phase += dist / 2.2;
        int guard = 0;
        while ((a->s > E->len || a->s < 0) && guard++ < 4) {
            int end_b = a->s > E->len;
            double over = end_b ? a->s - E->len : -a->s;
            int nd = end_b ? E->b : E->a;
            /* too many workers for the load: this one goes to rest (fades out) in a chamber */
            if (a->role == R_WORKER && !a->dying && (nodes[nd].rx > 20 || frand() < 0.25) &&
                count_role(R_WORKER, 0) > worker_target(d->cpu) + 1)
                a->dying = 1;
            int ne = choose_next(a, nd, a->e, d);
            if (ne < 0) {
                a->s = end_b ? E->len : 0;
                if (a->role == R_WORKER) {
                    /* dead end or linger: work a moment, then turn back */
                    a->pause = 0.3 + frand() * 0.8 * (1 - 0.6 * d->cpu);
                    a->reverse_after = 1;
                } else {
                    arrive_end(a, d);
                }
                break;
            }
            a->e = ne;
            E = &edges[ne];
            if (E->a == nd)
                a->dir = 1, a->s = over;
            else
                a->dir = -1, a->s = E->len - over;
        }
        if (a->role == R_NET_OUT && a->alive) {
            double x, y, tx, ty;
            edge_at(E, a->s, &x, &y, &tx, &ty);
            if (hypot(x - CX, y - CY) > 250)
                a->alive = 0;
        }
    }
    if (!a->alive)
        return;

    /* eased drawn position and heading */
    double x, y, tx, ty;
    edge_at(&edges[a->e], a->s, &x, &y, &tx, &ty);
    double hx = tx * a->dir, hy = ty * a->dir;
    double lane = edges[a->e].surf ? 0 : a->lane;
    x += -hy * lane;
    y += hx * lane;
    double target = atan2(hy, hx);
    if (a->pause > 0 && a->reverse_after && a->role == R_WORKER)
        target += M_PI * clamp01(1 - a->pause / 0.5);           /* turning round */
    double wob = d->frantic * 0.45 * sin(t * 11 + a->seed * 3) + 0.12 * sin(t * 2.3 + a->seed);
    if (a->x < -1e8) {
        a->x = x, a->y = y, a->ang = target;
    } else {
        double kp = fmin(1, dt * 14);
        a->x += (x - a->x) * kp;
        a->y += (y - a->y) * kp;
        double da = wrap_pi(target + wob * (a->pause > 0 ? 0.3 : 1) - a->ang);
        double maxr = (6 + 6 * d->frantic) * dt;
        a->ang += fmax(-maxr, fmin(maxr, da * fmin(1, dt * 10)));
    }
}

static void spawn_worker(int n)
{
    static const int homes[] = { NUR0, NUR1, FOOD, JC, NUR0, NUR1 };
    ant *a = new_ant(R_WORKER);
    if (!a)
        return;
    if (n < 0)
        n = homes[rand() % 6];
    int e = node_edges[n][rand() % node_deg[n]];
    if (edges[e].restricted)
        e = node_edges[n][0];
    a->e = e;
    a->dir = edges[e].a == n ? 1 : -1;
    a->s = a->dir > 0 ? frand() * 6 : edges[e].len - frand() * 6;
}

static void spawn_routed(int role, int from, int to, int carry, int disk)
{
    ant *a = new_ant(role);
    if (!a)
        return;
    a->nroute = find_route(from, to, a->route, 32);
    if (a->nroute < 2) {
        a->alive = 0;
        return;
    }
    a->carry = carry;
    a->disk = disk;
    if (role == R_NET_IN)
        a->alpha = 1;                   /* walks in from beyond the glass */
    start_route(a);
}

static void simulate(const drive_t *d, double dt, double t)
{
    static double acc_in, acc_out, acc_work;
    double cpu = d->cpu;

    /* workers: population follows CPU load; born in the nurseries, rest in chambers */
    int want = worker_target(cpu);
    int have = count_role(R_WORKER, 0);
    acc_work += dt * (have < want - 2 ? 1 + (want - have) * 0.4 : 0);
    while (acc_work >= 1) {
        spawn_worker(-1);
        acc_work -= 1;
    }
    if (have >= want)
        acc_work = 0;

    /* network: leaves carried in, ants heading out */
    double rxl = net_level(d->rx), txl = net_level(d->tx);
    acc_in += dt * (rxl > 0.05 ? 0.15 + 2.6 * rxl * rxl : 0);
    acc_out += dt * (txl > 0.05 ? 0.15 + 2.6 * txl * txl : 0);
    while (acc_in >= 1) {
        spawn_routed(R_NET_IN, rand() & 1 ? S_L : S_R, FOOD, IT_LEAF, 0);
        acc_in -= 1;
    }
    while (acc_out >= 1) {
        spawn_routed(R_NET_OUT, FOOD, rand() & 1 ? S_L : S_R, rand() % 3 == 0 ? IT_PEBBLE : IT_NONE, 0);
        acc_out -= 1;
    }

    /* haulers per drive, and to the swap cellar */
    for (int k = 0; k < N_STORES; k++) {
        static double acc_h[N_STORES];
        double up_l, dn_l;
        store_levels(d, k, &up_l, &dn_l);
        acc_h[k] += dt * (count_role(R_HAULER, k) < store_target(d, k) ? 0.9 : 0);
        if (acc_h[k] >= 1) {
            acc_h[k] = 0;
            spawn_routed(R_HAULER, FOOD, STORE_NODE[k], dn_l > 0.02 && frand() < dn_l / fmax(up_l, dn_l) ? store_item(k, 1) : IT_NONE, k);
        }
    }

    /* pressure stalls: the more time tasks spend waiting, the more the tunnels jam */
    double jam_cpu = clamp01(d->psi_cpu / PSI_FULL), jam_io = clamp01(d->psi_io / PSI_FULL);
    static double gap[MAX_ANTS];
    for (int i = 0; i < MAX_ANTS; i++) {
        gap[i] = 1e9;
        const ant *a = &ants[i];
        if (!a->alive)
            continue;
        for (int j = 0; j < MAX_ANTS; j++) {
            const ant *b = &ants[j];
            if (j == i || !b->alive || b->e != a->e || b->dir != a->dir || b->dying)
                continue;
            double g = (b->s - a->s) * a->dir;
            if (g > 0 && g < gap[i])
                gap[i] = g;
        }
    }

    for (int i = 0; i < MAX_ANTS; i++) {
        ant *a = &ants[i];
        if (!a->alive)
            continue;
        double v;
        switch (a->role) {
        case R_WORKER:  v = 13 + 44 * cpu; break;
        case R_HAULER: {
            double up_l, dn_l;
            store_levels(d, a->disk, &up_l, &dn_l);
            v = 22 + 40 * fmax(up_l, dn_l);
            break;
        }
        default:        v = 22 + 30 * fmax(rxl, txl); break;
        }
        v *= a->speed * (1 + 0.55 * d->frantic);
        double jam = a->role == R_HAULER ? jam_io : a->role == R_WORKER ? jam_cpu : fmax(jam_cpu, jam_io) * 0.5;
        if (jam > 0.02 && !edges[a->e].surf) {
            /* stop-and-go: leaders stall now and then, the ones behind queue up */
            if (a->pause <= 0 && frand() < dt * 0.6 * jam)
                a->pause = 0.3 + frand() * 1.2 * jam;
            double min_gap = 4 + 12 * jam;
            v *= clamp01((gap[i] - min_gap) / (min_gap * 0.8)) * (1 - 0.35 * jam);
        }
        a->v += (v - a->v) * fmin(1, dt * 8);
        ant_step(a, dt, a->v, d, t);
    }

    /* eggs: the queen lays one for every burst of new processes; they hatch into workers */
    static double acc_egg;
    double fl = clamp01(log10(1 + d->forks) / log10(1 + FORK_FULL));
    acc_egg += dt * (d->forks > 1 ? 0.15 + 2.4 * fl * fl : 0);
    for (int i = 0; i < 24; i++)
        if (eggs[i].life > 0) {
            eggs[i].age += dt;
            eggs[i].x += eggs[i].vx * dt * clamp01(1.5 - eggs[i].age);
            if (eggs[i].age > eggs[i].life) {
                eggs[i].life = 0;
                if (count_role(R_WORKER, 0) < want + 6)
                    spawn_worker(QUEEN);
            }
        }
    while (acc_egg >= 1) {
        acc_egg -= 1;
        for (int i = 0; i < 24; i++)
            if (eggs[i].life <= 0) {
                eggs[i] = (egg_t){ nodes[QUEEN].x + 36 + frand() * 4, nodes[QUEEN].y + nodes[QUEEN].oy + 12 + frand() * 6, 0, 4 + frand() * 4, 5 + frand() * 8 };
                break;
            }
    }
}

/* ---------------------------------------------------------------- render */

static void blit(cairo_t *cr, cairo_surface_t *spr, double x, double y, double alpha)
{
    double w = cairo_image_surface_get_width(spr), h = cairo_image_surface_get_height(spr);
    cairo_set_source_surface(cr, spr, x - w / 2, y - h / 2);
    if (alpha >= 0.999)
        cairo_paint(cr);
    else
        cairo_paint_with_alpha(cr, alpha);
}

static void glow(cairo_t *cr, cairo_surface_t *spr, double x, double y, double alpha)
{
    if (alpha < 0.01)
        return;
    double w = cairo_image_surface_get_width(spr), h = cairo_image_surface_get_height(spr);
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_set_source_surface(cr, spr, x - w / 2, y - h / 2);
    cairo_rectangle(cr, x - w / 2, y - h / 2, w, h);
    cairo_clip(cr);
    for (; alpha > 0.01; alpha -= 1)               /* above 1: add it again */
        cairo_paint_with_alpha(cr, fmin(1, alpha));
    cairo_reset_clip(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

static void draw_ant_at(cairo_t *cr, double x, double y, double ang, double phase, int pose, double alpha)
{
    int ai = (int)lround(ang / (2 * M_PI) * N_ANG) % N_ANG;
    if (ai < 0)
        ai += N_ANG;
    int f = ((int)floor(phase)) & 3;
    blit(cr, ant_spr[pose][ai][f], x, y, alpha);
}

static void soft_text(cairo_t *cr, double x, double y, double size, rgb c, double a, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    cairo_text_path(cr, s);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, size * 0.2);
    cairo_set_source_rgba(cr, 0.04, 0.02, 0.03, 0.8 * a);
    cairo_stroke_preserve(cr);
    cairo_set_source_rgba(cr, c.r, c.g, c.b, a);
    cairo_fill(cr);
}

/* text is shown with hysteresis so digits don't flicker between two values */
static double hyst(double shown, double v, double th)
{
    return fabs(v - shown) > th ? round(v) : shown;
}

static void update_hud(const drive_t *d, const sys_stats *ss, double t)
{
    static double cpu = -9, temp = -9, ram = -9, next;
    char key[128], txt[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        cpu = hyst(cpu, d->cpu * 100, 0.6);
        temp = hyst(temp, d->temp, 0.6);
        ram = hyst(ram, d->ram_used * ss->mem_total, 0.6);
    }
    snprintf(key, sizeof(key), "%.0f|%.0f|%.0f", cpu, temp, ram);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);

    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    rgb cream = { 1.0, 0.94, 0.84 }, dim = { 0.86, 0.72, 0.62 };
    rgb seedc = { 1.0, 0.80, 0.44 };
    rgb tc = lerp((rgb){ 1.0, 0.86, 0.6 }, (rgb){ 1.0, 0.36, 0.2 }, clamp01((temp - 60) / 30));
    soft_text(cr, 136, 62, 22, dim, 0.9, "RAM");
    soft_text(cr, CX, 50, 22, dim, 0.9, "CPU");
    soft_text(cr, 344, 62, 22, dim, 0.9, "TEMP");
    snprintf(txt, sizeof(txt), "%.0f%%", cpu);
    soft_text(cr, CX, 86, 42, cream, 1, txt);
    snprintf(txt, sizeof(txt), "%.0fG", ram);
    soft_text(cr, 136, 92, 28, seedc, 1, txt);
    snprintf(txt, sizeof(txt), "%.0f\xC2\xB0", temp);
    soft_text(cr, 344, 92, 28, tc, 1, txt);
    cairo_destroy(cr);
}

/* the stand's plaque: colony population = every task on the machine */
static void update_plaque(const drive_t *d, double t)
{
    static double tasks = -99, next;
    char key[32], num[24], txt[40];
    if (t >= next || t < next - 2) {
        next = t + 1;
        tasks = hyst(tasks, d->tasks, 2.5);
    }
    snprintf(key, sizeof(key), "%.0f", tasks);
    if (!strcmp(key, plaque_key))
        return;
    strcpy(plaque_key, key);
    long n = lround(fmax(0, tasks));
    if (n >= 1000)
        snprintf(num, sizeof(num), "%ld,%03ld", n / 1000, n % 1000);
    else
        snprintf(num, sizeof(num), "%ld", n);
    snprintf(txt, sizeof(txt), "%s ants", num);

    cairo_t *cr = cairo_create(plaque_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Serif Condensed", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 22);
    cairo_text_extents(cr, txt, &ext);
    double x = 75 - ext.width / 2 - ext.x_bearing, y = 14 - ext.height / 2 - ext.y_bearing;
    cairo_move_to(cr, x, y + 1);                    /* engraved: light lower edge, dark letters */
    cairo_text_path(cr, txt);
    cairo_set_source_rgba(cr, 1, 0.93, 0.7, 0.55);
    cairo_fill(cr);
    cairo_move_to(cr, x, y);
    cairo_text_path(cr, txt);
    cairo_set_source_rgb(cr, 0.24, 0.13, 0.04);
    cairo_fill(cr);
    cairo_destroy(cr);
}

static void render(cairo_t *cr, const drive_t *d, const sys_stats *ss, double t)
{
    /* background: cool/hot soil blended, re-made only when the tint moves a step */
    int step = (int)lround(d->heat * 40);
    if (step != bg_mix_step) {
        bg_mix_step = step;
        cairo_t *c2 = cairo_create(bg_mix);
        cairo_set_source_surface(c2, bg_cool, 0, 0);
        cairo_paint(c2);
        cairo_set_source_surface(c2, bg_hot, 0, 0);
        cairo_paint_with_alpha(c2, step / 40.0);
        cairo_destroy(c2);
    }
    cairo_set_source_surface(cr, bg_mix, 0, 0);
    cairo_paint(cr);

    /* food store: seeds (RAM used) with pale crumbs (page cache) on top */
    int nu = (int)lround(clamp01(d->ram_used) * FOOD_SEEDS / 4) * 4;
    int nc = (int)lround(clamp01(d->ram_cache) * FOOD_SEEDS / 4) * 4;
    static double food_next;
    if ((nu != ram_shown_used || nc != ram_shown_cache) && (t >= food_next || t < food_next - 1)) {
        food_next = t + 0.25;
        ram_shown_used = nu;
        ram_shown_cache = nc;
        draw_food(nu, nc);
    }
    const node *fn = &nodes[FOOD];
    glow(cr, glow_food, fn->x, fn->y, 0.35 + 0.4 * d->ram_used);
    cairo_set_source_surface(cr, food_layer, fn->x - fn->rx - 4, fn->y - fn->ry - 4);
    cairo_paint(cr);

    /* burrows along the surface: open TCP connections */
    double want_b = N_BURROWS * clamp01(log(1 + d->tcp / 10) / log(1 + 1000 / 10.0));
    if (fabs(want_b - burrows_shown) > 0.7)
        burrows_shown = (int)lround(want_b);
    for (int i = 0; i < N_BURROWS; i++) {
        burrow_a[i] = clamp01(burrow_a[i] + (i < burrows_shown ? 1 : -1) * (1.0 / FPS_BUSY) * 0.8);
        if (burrow_a[i] <= 0.01)
            continue;
        cairo_surface_t *b0 = burrow_spr[i][0], *b1 = burrow_spr[i][1];
        double bx = burrow_x[i] - 20, by = ground_y(burrow_x[i]) - 6;
        cairo_set_source_surface(cr, b0, bx, by);
        cairo_paint_with_alpha(cr, burrow_a[i]);
        if (d->heat > 0.02) {
            cairo_set_source_surface(cr, b1, bx, by);
            cairo_paint_with_alpha(cr, burrow_a[i] * d->heat);
        }
    }

    /* swap cellar: glows while memory moves between it and the store */
    {
        double up_l, dn_l;
        store_levels(d, SYS_NVME, &up_l, &dn_l);
        glow(cr, glow_cellar, nodes[CELLAR].x, nodes[CELLAR].y, 0.05 + 0.9 * fmax(up_l, dn_l));
    }

    /* nurseries: fungus glow and wriggling larvae, by each GPU's power */
    for (int k = 0; k < 2; k++) {
        const node *n = &nodes[NUR0 + k];
        double p = d->gpow[k], fl = 0.9 + 0.1 * sin(t * (2 + 5 * p) + k * 2);
        glow(cr, glow_nur[k], n->x, n->y + 6, (0.12 + 1.5 * p) * fl);
        for (int i = 0; i < 5; i++) {
            double lx = n->x - 26 + i * 13 + 2 * sin(i * 7.1), ly = n->y + 2 + 4 * sin(i * 2.3);
            double wig = sin(t * (1.5 + 6 * p) + i * 1.7) * (0.6 + 1.6 * p);
            blit(cr, larva_spr, lx + wig * 0.4, ly - fabs(wig) * 0.3, 0.9);
        }
    }

    /* queen's chamber: pulses with overall GPU activity */
    const node *qn = &nodes[QUEEN];
    double qy = qn->y + qn->oy + 6;
    double pulse = 0.5 + 0.5 * sin(t * (1.2 + 5 * d->gall));
    glow(cr, glow_queen, qn->x, qy - 6, 0.15 + d->gall * (0.45 + 0.5 * pulse));
    for (int i = 0; i < 24; i++)
        if (eggs[i].life > 0)
            blit(cr, egg_spr, eggs[i].x, eggs[i].y, clamp01(eggs[i].age * 3) * clamp01(eggs[i].life - eggs[i].age));
    blit(cr, queen_spr, qn->x - 4 + sin(t * 0.7) * 0.8, qy + d->gall * pulse * 0.6, 1);
    /* attendants grooming her */
    int n_att = 1 + (int)(d->gall * 3.5);
    static const double att[4][3] = { { -57, 0, 0.05 }, { -41, -15, 0.7 }, { -41, 14, -0.7 }, { 22, -18, 1.75 } };
    for (int i = 0; i < n_att && i < 4; i++) {
        double jig = sin(t * (3 + 8 * d->gall) + i * 2.1);
        draw_ant_at(cr, qn->x + att[i][0] + jig * 0.8, qy + att[i][1], att[i][2] + jig * 0.25,
                    t * (2 + 10 * d->gall) + i, d->frantic > 0.5 && ((int)(t * 8 + i) & 1), 1);
    }

    /* brood cells: one per core, glowing and bustling with its load */
    for (int i = 0; i < N_CELLS; i++) {
        const node *n = &nodes[CELL0 + i];
        double l = d->cell[i];
        glow(cr, glow_cell, n->x, n->y, 0.06 + 0.9 * l);
        double sp = 1.5 + 14 * l;
        int nres = l > 0.55 ? 2 : l > 0.12 ? 1 : 0;
        for (int r = 0; r < nres; r++) {
            double jig = sin(t * sp * 0.4 + i + r * 2);
            double ang = cell_ang[i] + (r ? M_PI + 0.3 : 0.2) + jig * 0.35 * l;
            draw_ant_at(cr, n->x + (r ? 5 : -4) * cos(cell_ang[i]) + jig * l, n->y + (r ? 2 : -1),
                        ang, t * sp + i * 1.3 + r, d->frantic > 0.5 && ((int)(t * 8 + i + r) & 1), 1);
        }
    }

    /* drive chambers: light by throughput, warm with the drive's temperature */
    for (int k = 0; k < SYS_NVME; k++) {
        const node *n = &nodes[NV0 + k];
        double lv = fmax(disk_level(d->rd[k]), disk_level(d->wr[k]));
        glow(cr, glow_nv, n->x, n->y, 0.08 + 0.9 * lv);
    }

    /* the ants */
    for (int i = 0; i < MAX_ANTS; i++) {
        const ant *a = &ants[i];
        if (!a->alive || a->alpha <= 0.01)
            continue;
        double al = a->alpha;
        /* fade into the shaft mouth / out beyond the glass */
        if (a->role == R_NET_IN || a->role == R_NET_OUT) {
            double r = hypot(a->x - CX, a->y - CY);
            al *= clamp01((R_GLASS + 6 - r) / 14);
        }
        int pose = d->frantic > 0.5 && ((int)(t * 8 + a->seed) & 1);
        draw_ant_at(cr, a->x, a->y, a->ang, a->phase, pose, al);
        if (a->carry != IT_NONE)
            blit(cr, item_spr[a->carry], a->x + cos(a->ang) * 10.5, a->y + sin(a->ang) * 10.5, al);
    }

    cairo_set_source_surface(cr, fg_strip, 0, 100);
    cairo_paint(cr);
    cairo_set_source_surface(cr, glass, 0, 0);
    cairo_paint(cr);

    update_hud(d, ss, t);
    cairo_set_source_surface(cr, hud_cache, 0, 0);
    cairo_paint(cr);
    update_plaque(d, t);
    cairo_set_source_surface(cr, plaque_cache, CX - 75, BASE_Y + 6);
    cairo_paint(cr);
}

/* ---------------------------------------------------------------- inputs */

/* from raw readings to the eased values the scene uses */
static void update_drive(drive_t *d, const sys_stats *s, const stats *g, double dt)
{
    double k = fmin(1, dt * 1.6), kc = fmin(1, dt * 3);
    d->cpu += (s->cpu_total - d->cpu) * kc;
    for (int i = 0; i < N_CELLS; i++) {
        double v = s->cpu[i];
        if (s->ncpu > N_CELLS)
            v = 0.5 * (s->cpu[i] + s->cpu[(i + N_CELLS) % SYS_MAX_CPU]);   /* both SMT threads of core i */
        d->cell[i] += (v - d->cell[i]) * kc;
    }
    if (s->cpu_temp > 0) {
        if (d->temp <= 0)
            d->temp = s->cpu_temp;
        d->temp += (s->cpu_temp - d->temp) * k;
    }
    d->heat = clamp01((d->temp - TEMP_COOL) / (TEMP_HOT - TEMP_COOL));
    if (d->temp > TEMP_FRANTIC)
        frantic_on = 1;
    else if (d->temp < TEMP_FRANTIC - 4)
        frantic_on = 0;
    d->frantic += (frantic_on - d->frantic) * fmin(1, dt * 1.2);
    if (s->mem_total > 0) {
        d->ram_used += (s->mem_used / s->mem_total - d->ram_used) * k;
        d->ram_cache += (fmin(s->mem_cached, s->mem_total - s->mem_used) / s->mem_total - d->ram_cache) * k;
    }
    for (int i = 0; i < SYS_NVME; i++) {
        d->rd[i] += (s->disk_rd[i] - d->rd[i]) * k;
        d->wr[i] += (s->disk_wr[i] - d->wr[i]) * k;
        d->nvtemp[i] += (s->nvme_temp[i] - d->nvtemp[i]) * k;
    }
    d->rx += (s->net_rx - d->rx) * k;
    d->tx += (s->net_tx - d->tx) * k;
    d->swin += (s->swapin_s - d->swin) * k;
    d->swout += (s->swapout_s - d->swout) * k;
    d->forks += (s->forks_s - d->forks) * k;
    if (d->tasks <= 0)
        d->tasks = s->tasks;
    d->tasks += (s->tasks - d->tasks) * fmin(1, dt * 0.8);
    d->psi_cpu += (s->psi_cpu - d->psi_cpu) * k;
    d->psi_io += (s->psi_io - d->psi_io) * k;
    if (d->tcp <= 0)
        d->tcp = s->tcp_inuse;
    d->tcp += (s->tcp_inuse - d->tcp) * fmin(1, dt * 0.8);
    double all = 0;
    for (int i = 0; i < N_GPUS; i++) {
        double pw = clamp01((g->power[i] - GPU_IDLE_W) / (GPU_MAX_W - GPU_IDLE_W));
        double act = 0.5 * clamp01(g->load[i]) + 0.5 * pw;
        if (act < 0.03)
            act = 0;
        d->gpow[i] += (pw - d->gpow[i]) * k;
        d->gact[i] += (act - d->gact[i]) * k;
        all += act / N_GPUS;
    }
    /* vLLM bonus: generating tokens keeps the queen busy even at low GPU load */
    if (!gpu_source && g->tok_s > 1)
        all = fmax(all, clamp01(g->tok_s / 1500));
    d->gall += (all - d->gall) * k;
}

static double noise1(double t, double seed)
{
    return 0.5 + 0.25 * sin(t * 1.3 + seed) + 0.15 * sin(t * 3.1 + seed * 2.7) + 0.1 * sin(t * 7.7 + seed * 5.1);
}

/* --demo: everything wanders through varied ranges */
static void demo_sys(sys_stats *s, double t)
{
    double busy = 0.5 + 0.45 * sin(t * 0.21) + 0.1 * sin(t * 0.73);
    busy = clamp01(busy);
    s->ncpu = 32;
    double tot = 0;
    for (int i = 0; i < 32; i++) {
        double v = busy * (0.4 + 0.9 * noise1(t * 0.8, i * 1.37)) * (i % 16 < 8 ? 1.0 : 0.75);
        s->cpu[i] = clamp01(v + (frand() - 0.5) * 0.06);
        tot += s->cpu[i];
    }
    s->cpu_total = tot / 32;
    s->cpu_temp = 44 + 48 * pow(busy, 1.3);
    s->mem_total = 91;
    s->mem_used = 18 + 60 * (0.5 + 0.5 * sin(t * 0.09));
    s->mem_cached = 4 + 8 * (0.5 + 0.5 * sin(t * 0.13 + 1));
    for (int k = 0; k < 3; k++) {
        double ph = noise1(t * 0.35, k * 3.3);
        s->disk_rd[k] = ph > 0.55 ? pow(10, 5 + 4.2 * (ph - 0.55) / 0.45) : 0;
        s->disk_wr[k] = ph < 0.42 ? pow(10, 5 + 4.0 * (0.42 - ph) / 0.42) : 0;
        s->nvme_temp[k] = 38 + 20 * ph;
    }
    double n = noise1(t * 0.3, 9.1);
    s->net_rx = pow(10, 3 + 5 * n);
    s->net_tx = pow(10, 3 + 4.2 * noise1(t * 0.27, 4.4));
    s->forks_s = pow(10, 0.8 + 2.8 * busy * noise1(t * 0.5, 2.2) * 1.4);
    s->tasks = (int)(3700 + 900 * busy + 60 * sin(t * 0.4));
    s->procs_running = 1 + (int)(busy * 30);
    s->psi_cpu = busy > 0.7 ? (busy - 0.7) / 0.3 * 40 * noise1(t * 0.6, 7.7) : 0;
    s->psi_io = 30 * clamp01(noise1(t * 0.4, 3.9) * 2.4 - 1.3);
    double sw = clamp01(noise1(t * 0.23, 5.5) * 2.5 - 1.4);
    s->swapin_s = sw > 0 ? pow(10, 1 + 3.5 * sw) : 0;
    s->swapout_s = sw > 0.3 ? pow(10, 1 + 3 * sw) : 0;
    s->tcp_inuse = (int)(80 + 500 * pow(n, 3));
}

/* --showcase: a scripted 36 s arc from a quiet colony to everything maxed and back */
static void showcase_poll(sys_stats *s, stats *g, double t)
{
    double u = fmod(t, 36.0);
    double env = u < 3 ? 0 : u < 20 ? (u - 3) / 17 : u < 27 ? 1 : u < 34 ? 1 - (u - 27) / 7 : 0;
    env = env * env * (3 - 2 * env);
    double cpu = 0.04 + 0.94 * env;
    s->ncpu = 32;
    double tot = 0;
    for (int i = 0; i < 32; i++) {
        double lag = clamp01(env * 1.25 - (i % 16) * 0.018);
        double v = 0.03 + (cpu - 0.03) * 1.25 * (0.8 + 0.7 * (noise1(t * 0.9, i * 1.9) - 0.5)) * (0.4 + 0.6 * lag);
        s->cpu[i] = clamp01(v + (frand() - 0.5) * 0.04);
        tot += s->cpu[i];
    }
    s->cpu_total = tot / 32;
    s->cpu_temp = 46 + 50 * pow(env, 1.4);
    s->mem_total = 91;
    s->mem_used = 14 + 70 * clamp01((u - 5) / 16) * (u < 27 ? 1 : 1 - clamp01((u - 27) / 8) * 0.85);
    s->mem_cached = 3 + 9 * env;
    /* network burst early, disks mid-arc, GPUs join, then everything */
    double net = clamp01((u - 5) / 3) * (u < 27 ? 1 : 1 - clamp01((u - 27) / 5));
    s->net_rx = net > 0 ? pow(10, 3.5 + 5 * net * (0.8 + 0.2 * sin(u * 1.7))) : 400;
    s->net_tx = net > 0 ? pow(10, 3.2 + 4.3 * net * (0.8 + 0.2 * sin(u * 1.3 + 1))) : 300;
    for (int k = 0; k < 3; k++) {
        double dk = clamp01((u - 9 - k * 2.5) / 3) * (u < 27 ? 1 : 1 - clamp01((u - 27 + k) / 5));
        s->disk_wr[k] = dk > 0 ? pow(10, 5 + 4.4 * dk * (k == 1 ? 0.7 : 1)) : 0;
        s->disk_rd[k] = dk > 0 ? pow(10, 5 + 4.3 * dk * (k == 1 ? 1 : 0.6)) : 0;
        s->nvme_temp[k] = 40 + 25 * dk;
    }
    double g0 = clamp01((u - 12) / 3) * (u < 27 ? 1 : 1 - clamp01((u - 27) / 5));
    double g1 = clamp01((u - 16) / 3) * (u < 27 ? 1 : 1 - clamp01((u - 28) / 5));
    g->load[0] = g0, g->load[1] = g1;
    g->power[0] = 35 + 540 * g0 * (0.95 + 0.05 * sin(u * 2));
    g->power[1] = 40 + 530 * g1 * (0.95 + 0.05 * sin(u * 2.4));
    g->temp[0] = (int)(38 + 30 * g0);
    g->temp[1] = (int)(45 + 30 * g1);
    g->tok_s = 0;
    s->forks_s = pow(10, 1.2 + 2.6 * env);
    s->tasks = (int)(3760 + 1100 * env);
    s->procs_running = 1 + (int)(30 * env);
    s->psi_cpu = u > 20 && u < 29 ? 38 * clamp01((u - 20) / 2) * clamp01((29 - u) / 2) : 0;
    s->psi_io = u > 11 && u < 17 ? 28 * clamp01((u - 11) / 1.5) * clamp01((17 - u) / 1.5) : 0;
    double sw = u > 21 && u < 30 ? clamp01((u - 21) / 2) * clamp01((30 - u) / 2) : 0;
    s->swapout_s = sw > 0 ? pow(10, 2 + 2.6 * sw) : 0;
    s->swapin_s = u > 23 && u < 32 ? pow(10, 2 + 2.4 * clamp01((u - 23) / 2) * clamp01((32 - u) / 2)) : 0;
    s->tcp_inuse = (int)(85 + 700 * net * net);
}

/* ---------------------------------------------------------------- main */

static int is_idle(const drive_t *d)
{
    return d->cpu < 0.08 && d->gall < 0.03 && net_level(d->rx) < 0.3 && net_level(d->tx) < 0.3;
}

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats g = { 0 };
    sys_stats ss = { 0 };
    drive_t d = { 0 };
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
    build_caches();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* showcase moments: maxed, midway, quiet */
        struct { double at; const char *png; } scenes[] = {
            { 24, "antfarm_preview.png" }, { 13, "antfarm_mid.png" }, { 1.5, "antfarm_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(5 + k);
            memset(ants, 0, sizeof(ants));
            memset(&d, 0, sizeof(d));
            hud_key[0] = 0;
            ram_shown_used = ram_shown_cache = -1;
            double start = fmax(0, scenes[k].at - 12);
            int n = (int)((scenes[k].at - start) * FPS_BUSY);
            size_t len = 0;
            double b0 = 0;
            int tail = n < 60 ? n : 60;
            for (int i = 0; i < n; i++) {
                if (i == n - tail)
                    b0 = now_s();
                double t = start + i / (double)FPS_BUSY;
                showcase_poll(&ss, &g, t);
                update_drive(&d, &ss, &g, 1.0 / FPS_BUSY);
                simulate(&d, 1.0 / FPS_BUSY, t);
                render(cr, &d, &ss, t);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            int alive = 0;
            for (int i = 0; i < MAX_ANTS; i++)
                alive += ants[i].alive;
            printf("%s: %.2f ms/frame, jpeg %zu bytes, %d ants\n", scenes[k].png,
                   (now_s() - b0) * 1000 / tail, len, alive);
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
            showcase_poll(&ss, &g, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 0.5;
            if (demo) {
                /* simulate into a base copy each poll; never scale s itself */
                static stats demo_base;
                static sys_stats sys_base;
                demo_poll(&demo_base, t - t0);
                demo_sys(&sys_base, t - t0);
                g = demo_base;
                ss = sys_base;
                g.power[0] = demo_base.power[0] * 1.2;
                g.power[1] = demo_base.power[1] * 1.15;
            } else {
                gpus_poll(&g);
                sys_poll(&ss, t);
                if (getenv("ANTFARM_DEBUG"))
                    fprintf(stderr, "cpu %.0f%% %.1fC ram %.1f/%.1fG cache %.1fG | nvme r %.0f/%.0f/%.0f w %.0f/%.0f/%.0f KB/s %.0f/%.0f/%.0fC | "
                            "net %.0f/%.0f KB/s | forks %.0f/s tasks %d psi cpu %.1f io %.1f mem %.1f | swap in %.0f out %.0f | tcp %d | "
                            "dimm %.0f/%.0fC pkg %.0fW | gpu %.0f/%.0fW\n",
                            ss.cpu_total * 100, ss.cpu_temp, ss.mem_used, ss.mem_total, ss.mem_cached,
                            ss.disk_rd[0] / 1e3, ss.disk_rd[1] / 1e3, ss.disk_rd[2] / 1e3, ss.disk_wr[0] / 1e3, ss.disk_wr[1] / 1e3, ss.disk_wr[2] / 1e3,
                            ss.nvme_temp[0], ss.nvme_temp[1], ss.nvme_temp[2], ss.net_rx / 1e3, ss.net_tx / 1e3,
                            ss.forks_s, ss.tasks, ss.psi_cpu, ss.psi_io, ss.psi_mem, ss.swapin_s, ss.swapout_s, ss.tcp_inuse,
                            ss.dimm_temp[0], ss.dimm_temp[1], ss.cpu_watts, g.power[0], g.power[1]);
                if (gpu_source)
                    gpu_rate_poll(&g);
                else
                    vllm_poll(&g, t);
            }
        }

        update_drive(&d, &ss, &g, dt);
        simulate(&d, dt, t - t0);
        render(cr, &d, &ss, t - t0);
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

        double spare = 1.0 / (is_idle(&d) && !showcase ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
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
