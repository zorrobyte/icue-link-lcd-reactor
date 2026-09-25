/*
 * skyline: a city at night on the iCUE LINK AIO pump LCD, drawn from the whole
 * machine's sensors rather than just the GPUs.
 *
 * Every CPU thread is a building: the 16 physical cores are the front row, their 16
 * SMT siblings the taller towers behind, and each building's windows light up with
 * that thread's load. Headlights and taillights on the waterfront road are network
 * traffic in and out, trains on the viaduct are NVMe reads and writes, the moon's
 * phase is RAM in use, and the sky warms from deep night blue to smoggy orange as the
 * CPU heats up. The two GPUs are the power plants on the horizon, glowing and
 * steaming with their power draw (the steam rises faster with the GPU fans). Each
 * roof's neon shows its thread's clock, the traffic signals cycle with context
 * switches, taxis join the traffic with every burst of new processes, the radio mast
 * sends out a wave per burst of open TCP connections, and the glow over the city is
 * the CPU package power. When vLLM is generating, searchlights sweep the sky.
 * Everything is drawn with cairo; static layers are cached at startup.
 * Run with --demo to simulate data, --showcase for a scripted 36 s night,
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
    double tok_s;               /* all servers */
    double tok_port[2];         /* per server: [0] ZOTAC's vLLM, [1] TUF's vLLM */
    int    running;
    int    running_port[2];     /* running requests per server */
    double fan[N_GPUS];         /* 0..1 */
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
 * (skyline is driven by system sensors and GPU power either way; tok/s only adds the
 * searchlights, so in GPU mode they sweep with GPU activity.)
 */
#define GPU_FULL_RATE   900.0
#define GPU_IDLE_W      40.0        /* board power at idle */
#define GPU_MAX_W       575.0       /* board power limit */
static int gpu_source;

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
        unsigned int fan;
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
 * Whole-machine sensors, read straight from /proc and /sys (no libraries). Call
 * sys_init() once, then sys_poll() at most a couple of times a second: rates (CPU
 * load, disk and network throughput) are differenced between calls. hwmon devices
 * are found by name, never by number. Values that can't be read are left at 0
 * (temperatures at NAN in the raw reads, then skipped).
 */
#define MAX_CPUS    64
#define N_NVME      3
#define NET_IF      "enp12s0"
#define NET_IF_TS   "tailscale0"

typedef struct {
    int    ncpu;
    double cpu[MAX_CPUS];           /* per-thread load 0..1 */
    double cpu_avg;                 /* 0..1 */
    double cpu_temp;                /* k10temp Tctl, C */
    double cpu_pkg_temp, mb_temp, vrm_temp;     /* asusec, C */
    double ram_total, ram_used, ram_cached;     /* bytes; used = total - available */
    double ram_frac;                /* used / total */
    double dimm_temp[2];            /* spd5118, C */
    double nvme_temp[N_NVME];       /* C */
    double disk_rd[N_NVME], disk_wr[N_NVME];    /* bytes/s per drive (nvme0n1..nvme2n1) */
    double disk_read, disk_write;   /* bytes/s, all drives */
    double net_rx, net_tx;          /* bytes/s on NET_IF */
    double ts_rx, ts_tx;            /* bytes/s on NET_IF_TS */
    double freq[MAX_CPUS];          /* per-thread clock, MHz */
    double ctxt_s, intr_s, forks_s; /* context switches, interrupts, forks per second */
    int    procs_running;
    double load1;                   /* 1-minute load average */
    int    tcp_inuse;               /* open TCP sockets */
    double psi_cpu, psi_io, psi_mem;    /* PSI "some avg10", % of time stalled */
    double cpu_watts;               /* package power from RAPL; NAN when unreadable (needs root) */
} sys_stats;

static struct {
    char k10[320], cpu_pkg[320], mobo[320], vrm[320];
    char dimm[2][320], nvme[N_NVME][320];
    int  n_dimm, n_nvme;
} hw;

static int read_text(const char *path, char *buf, int cap)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC), n = 0;
    ssize_t r;
    if (fd < 0)
        return -1;
    while (n < cap - 1 && (r = read(fd, buf + n, cap - 1 - n)) > 0)
        n += (int)r;
    close(fd);
    buf[n] = 0;
    return n;
}

static double read_milli(const char *path)
{
    char b[32];
    if (!path[0] || read_text(path, b, sizeof(b)) <= 0)
        return NAN;
    return atof(b) / 1000.0;
}

static void sys_init(void)
{
    DIR *d = opendir("/sys/class/hwmon");
    struct dirent *e;
    char path[400], name[64], label[64];

    if (!d)
        return;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5) != 0)
            continue;
        snprintf(path, sizeof(path), "/sys/class/hwmon/%s/name", e->d_name);
        if (read_text(path, name, sizeof(name)) <= 0)
            continue;
        name[strcspn(name, "\n")] = 0;
#define HWPATH(dst, file) snprintf(dst, sizeof(dst), "/sys/class/hwmon/%s/%s", e->d_name, file)
        if (!strcmp(name, "k10temp")) {
            HWPATH(hw.k10, "temp1_input");                  /* Tctl */
        } else if (!strcmp(name, "asusec")) {
            for (int i = 1; i <= 8; i++) {
                char f[32];
                snprintf(f, sizeof(f), "temp%d_label", i);
                HWPATH(path, f);
                if (read_text(path, label, sizeof(label)) <= 0)
                    continue;
                label[strcspn(label, "\n")] = 0;
                snprintf(f, sizeof(f), "temp%d_input", i);
                if (!strcmp(label, "CPU Package"))
                    HWPATH(hw.cpu_pkg, f);
                else if (!strcmp(label, "Motherboard"))
                    HWPATH(hw.mobo, f);
                else if (!strcmp(label, "VRM"))
                    HWPATH(hw.vrm, f);
            }
        } else if (!strcmp(name, "spd5118") && hw.n_dimm < 2) {
            HWPATH(hw.dimm[hw.n_dimm], "temp1_input");
            hw.n_dimm++;
        } else if (!strcmp(name, "nvme") && hw.n_nvme < N_NVME) {
            HWPATH(hw.nvme[hw.n_nvme], "temp1_input");
            hw.n_nvme++;
        }
#undef HWPATH
    }
    closedir(d);
}

static double temp_or(double v, double fallback) { return isnan(v) ? fallback : v; }

static void sys_poll(sys_stats *y, double t)
{
    static char buf[32768];
    static unsigned long long cpu_busy[MAX_CPUS], cpu_total[MAX_CPUS];
    static unsigned long long disk_r[N_NVME], disk_w[N_NVME], net[4];
    static unsigned long long ctxt0, intr0, forks0;
    static double last_t;
    static int have;
    double dt = t - last_t;
    char *p;

    if (dt <= 0.05)
        dt = 0.05;

    /* CPU load per thread */
    if (read_text("/proc/stat", buf, sizeof(buf)) > 0) {
        int n = 0;
        double sum = 0;
        for (p = strstr(buf, "\ncpu"); p; p = strstr(p + 1, "\ncpu")) {
            unsigned long long v[8] = { 0 };
            char *q = p + 4;
            if (*q < '0' || *q > '9')
                continue;
            int id = (int)strtol(q, &q, 10);
            if (id < 0 || id >= MAX_CPUS)
                continue;
            for (int k = 0; k < 8; k++)
                v[k] = strtoull(q, &q, 10);
            unsigned long long total = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
            unsigned long long busy = total - v[3] - v[4];         /* minus idle and iowait */
            if (have && total > cpu_total[id])
                y->cpu[id] = clamp01((double)(busy - cpu_busy[id]) / (double)(total - cpu_total[id]));
            cpu_busy[id] = busy;
            cpu_total[id] = total;
            sum += y->cpu[id];
            if (id + 1 > n)
                n = id + 1;
        }
        y->ncpu = n;
        y->cpu_avg = n ? sum / n : 0;
        {        /* scheduler counters, same read */
            unsigned long long c = 0, in = 0, f = 0;
            if ((p = strstr(buf, "\nctxt ")))
                c = strtoull(p + 6, NULL, 10);
            if ((p = strstr(buf, "\nintr ")))
                in = strtoull(p + 6, NULL, 10);
            if ((p = strstr(buf, "\nprocesses ")))
                f = strtoull(p + 11, NULL, 10);
            if ((p = strstr(buf, "\nprocs_running ")))
                y->procs_running = atoi(p + 15);
            if (have && c >= ctxt0 && in >= intr0 && f >= forks0) {
                y->ctxt_s = (c - ctxt0) / dt;
                y->intr_s = (in - intr0) / dt;
                y->forks_s = (f - forks0) / dt;
            }
            ctxt0 = c, intr0 = in, forks0 = f;
        }
    }

    /* temperatures */
    y->cpu_temp = temp_or(read_milli(hw.k10), y->cpu_temp);
    y->cpu_pkg_temp = temp_or(read_milli(hw.cpu_pkg), y->cpu_pkg_temp);
    y->mb_temp = temp_or(read_milli(hw.mobo), y->mb_temp);
    y->vrm_temp = temp_or(read_milli(hw.vrm), y->vrm_temp);
    for (int i = 0; i < hw.n_dimm; i++)
        y->dimm_temp[i] = temp_or(read_milli(hw.dimm[i]), y->dimm_temp[i]);
    for (int i = 0; i < hw.n_nvme; i++)
        y->nvme_temp[i] = temp_or(read_milli(hw.nvme[i]), y->nvme_temp[i]);

    /* memory */
    if (read_text("/proc/meminfo", buf, sizeof(buf)) > 0) {
        double total = 0, avail = 0, cached = 0;
        if ((p = strstr(buf, "MemTotal:")))
            total = strtod(p + 9, NULL) * 1024;
        if ((p = strstr(buf, "MemAvailable:")))
            avail = strtod(p + 13, NULL) * 1024;
        if ((p = strstr(buf, "\nCached:")))
            cached = strtod(p + 8, NULL) * 1024;
        if (total > 0) {
            y->ram_total = total;
            y->ram_used = total - avail;
            y->ram_cached = cached;
            y->ram_frac = clamp01(y->ram_used / total);
        }
    }

    /* NVMe throughput: sectors read (field 6) and written (field 10), 512 bytes each */
    if (read_text("/proc/diskstats", buf, sizeof(buf)) > 0) {
        y->disk_read = y->disk_write = 0;
        for (int i = 0; i < N_NVME; i++) {
            char key[24];
            snprintf(key, sizeof(key), " nvme%dn1 ", i);
            if (!(p = strstr(buf, key)))
                continue;
            unsigned long long f[7];
            char *q = p + strlen(key);
            for (int k = 0; k < 7; k++)
                f[k] = strtoull(q, &q, 10);                 /* fields 4..10 */
            if (have && f[2] >= disk_r[i] && f[6] >= disk_w[i]) {
                y->disk_rd[i] = (f[2] - disk_r[i]) * 512.0 / dt;
                y->disk_wr[i] = (f[6] - disk_w[i]) * 512.0 / dt;
            }
            disk_r[i] = f[2];
            disk_w[i] = f[6];
            y->disk_read += y->disk_rd[i];
            y->disk_write += y->disk_wr[i];
        }
    }

    /* network: rx bytes is the 1st field after the colon, tx bytes the 9th */
    if (read_text("/proc/net/dev", buf, sizeof(buf)) > 0) {
        static const char *ifs[2] = { NET_IF ":", NET_IF_TS ":" };
        double *rx[2] = { &y->net_rx, &y->ts_rx }, *tx[2] = { &y->net_tx, &y->ts_tx };
        for (int i = 0; i < 2; i++) {
            if (!(p = strstr(buf, ifs[i])))
                continue;
            char *q = p + strlen(ifs[i]);
            unsigned long long f[9];
            for (int k = 0; k < 9; k++)
                f[k] = strtoull(q, &q, 10);
            if (have && f[0] >= net[i * 2] && f[8] >= net[i * 2 + 1]) {
                *rx[i] = (f[0] - net[i * 2]) / dt;
                *tx[i] = (f[8] - net[i * 2 + 1]) / dt;
            }
            net[i * 2] = f[0];
            net[i * 2 + 1] = f[8];
        }
    }
    /* per-thread clocks */
    for (int i = 0; i < y->ncpu && i < MAX_CPUS; i++) {
        char path[96], b[32];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        if (read_text(path, b, sizeof(b)) > 0)
            y->freq[i] = atof(b) / 1000.0;
    }

    if (read_text("/proc/loadavg", buf, sizeof(buf)) > 0)
        y->load1 = atof(buf);
    if (read_text("/proc/net/sockstat", buf, sizeof(buf)) > 0 && (p = strstr(buf, "TCP: inuse ")))
        y->tcp_inuse = atoi(p + 11);
    static const char *psi[3] = { "/proc/pressure/cpu", "/proc/pressure/io", "/proc/pressure/memory" };
    double *psi_v[3] = { &y->psi_cpu, &y->psi_io, &y->psi_mem };
    for (int i = 0; i < 3; i++)
        if (read_text(psi[i], buf, sizeof(buf)) > 0 && (p = strstr(buf, "some avg10=")))
            *psi_v[i] = atof(p + 11);

    /* CPU package power: RAPL energy counter, root only; stays NAN when we can't read it */
    static unsigned long long e0;
    static int rapl = 1;
    y->cpu_watts = NAN;
    if (rapl) {
        char b[32];
        if (read_text("/sys/class/powercap/intel-rapl:0/energy_uj", b, sizeof(b)) > 0) {
            unsigned long long e = strtoull(b, NULL, 10);
            if (have && e0 && e >= e0)
                y->cpu_watts = (e - e0) / 1e6 / dt;
            e0 = e;
        } else {
            rapl = 0;                   /* EACCES or missing: don't keep trying */
        }
    }

    last_t = t;
    have = 1;
}

/* ---- end system sensors ---- */

static double frand(void) { return rand() / (double)RAND_MAX; }
static double smooth(double a, double b, double x) { x = clamp01((x - a) / (b - a)); return x * x * (3 - 2 * x); }

/* Log scale 0..1 for a throughput between lo and hi bytes/s */
static double logn(double v, double lo, double hi) { return v <= lo ? 0 : clamp01(log(v / lo) / log(hi / lo)); }

/*
 * --demo: a busy machine drifting through light and heavy spells. Every value is
 * computed fresh from t each poll (nothing is scaled in place, so nothing compounds).
 */
static void demo_poll(sys_stats *y, stats *s, double t)
{
    double busy = clamp01(0.5 + 0.42 * sin(t * 0.19) + 0.18 * sin(t * 0.53 + 1));
    y->ncpu = 32;
    y->cpu_avg = 0;
    for (int i = 0; i < 32; i++) {
        double own = 0.5 + 0.5 * sin(t * (0.3 + 0.07 * (i % 7)) + i * 1.7);
        double v = busy * (0.35 + 0.75 * own) + (frand() - 0.5) * 0.08;
        if (i >= 16)
            v *= 0.55 + 0.4 * busy;                         /* SMT siblings a little lighter */
        y->cpu[i] = clamp01(v);
        y->cpu_avg += y->cpu[i] / 32;
    }
    y->cpu_temp = 44 + 46 * pow(busy, 1.3);
    y->ram_frac = clamp01(0.35 + 0.45 * (0.5 + 0.5 * sin(t * 0.07)));
    y->ram_total = 96454912.0 * 1024;
    y->ram_used = y->ram_frac * y->ram_total;
    double net = 0.5 + 0.5 * sin(t * 0.31 + 2);
    y->net_rx = 2e4 * pow(10, 4.2 * net * (0.6 + 0.4 * busy));
    y->net_tx = 1e4 * pow(10, 3.6 * (0.5 + 0.5 * sin(t * 0.23)));
    double rd = sin(t * 0.17 + 0.5), wr = sin(t * 0.13 + 3);
    y->disk_read = rd > 0 ? 3e7 * pow(10, 2.3 * rd) : 0;
    y->disk_write = wr > 0.2 ? 2e7 * pow(10, 2.2 * (wr - 0.2)) : 0;
    for (int i = 0; i < N_GPUS; i++) {
        double g = clamp01(0.5 + 0.55 * sin(t * (0.11 + 0.05 * i) + i * 2.2));
        s->power[i] = 35 + 530 * g;
        s->load[i] = clamp01(g * 1.2);
        s->temp[i] = (int)(34 + 40 * g);
    }
    for (int i = 0; i < 32; i++)
        y->freq[i] = 600 + 5100 * clamp01(0.25 + 0.8 * y->cpu[i] + 0.2 * sin(t * 0.4 + i * 2.3));
    y->ctxt_s = 2e4 * pow(10, 2.1 * busy);
    double fk = sin(t * 0.15 + 1.3);
    y->forks_s = 5 + (fk > 0.2 ? 3000 * pow(fk - 0.2, 2) : 0);
    y->tcp_inuse = (int)(60 + 700 * net * net);
    y->cpu_watts = 40 + 170 * pow(busy, 1.2);
    y->load1 = 32 * busy;
    for (int i = 0; i < N_GPUS; i++)
        s->fan[i] = 0.3 + 0.65 * clamp01((s->power[i] - 35) / 530);
    s->tok_port[0] = s->tok_port[1] = s->tok_s = 0;
    s->running = 0;
    if (sin(t * 0.09) > 0.3) {                              /* now and then, vLLM is serving */
        s->tok_port[0] = 400 * s->load[0];
        s->tok_port[1] = 350 * s->load[1];
        s->tok_s = s->tok_port[0] + s->tok_port[1];
        s->running = 2;
    }
}

/*
 * --showcase: a scripted 36 s arc for filming: a sleeping city, a download and a
 * model load (traffic, then read trains), the CPU and both GPUs flat out while the
 * sky turns to smog and checkpoints are written, then back to sleep.
 */
#define SHOWCASE_LEN 36.0
static void showcase_poll(sys_stats *y, stats *s, double t)
{
    double u = fmod(t, SHOWCASE_LEN);
    double I = 0.03 + 0.42 * smooth(5, 10, u) + 0.55 * smooth(11, 15, u);    /* CPU intensity */
    I -= 0.94 * smooth(23, 30, u);
    I = clamp01(I);
    double G = smooth(8, 13, u) * (1 - smooth(24, 30, u));                  /* GPUs */
    double N = 0.1 + 0.9 * smooth(3, 6, u) * (1 - 0.6 * smooth(12, 16, u)) * (1 - smooth(25, 30, u));
    double R = smooth(6, 8, u) * (1 - smooth(12, 14, u)) + 0.5 * smooth(15, 16, u) * (1 - smooth(17, 18.5, u));
    double W = smooth(17, 18.5, u) * (1 - smooth(24, 26, u));

    y->ncpu = 32;
    y->cpu_avg = 0;
    for (int i = 0; i < 32; i++) {
        double own = 0.5 + 0.5 * sin(u * (0.5 + 0.09 * (i % 5)) + i * 2.1);
        double v = I < 0.6 ? I * (0.3 + 1.4 * own) : I * (0.9 + 0.1 * own);
        if (i >= 16)
            v *= 0.75 + 0.25 * I;
        v += (frand() - 0.5) * 0.04;
        y->cpu[i] = clamp01(v + 0.02);
        y->cpu_avg += y->cpu[i] / 32;
    }
    y->cpu_temp = 42 + 52 * pow(I, 1.2);
    y->ram_frac = 0.18 + 0.30 * smooth(6, 12, u) + 0.40 * smooth(12, 18, u) - 0.62 * smooth(26, 32, u);
    y->ram_total = 96454912.0 * 1024;
    y->ram_used = y->ram_frac * y->ram_total;
    y->net_rx = 3e3 + 3e8 * pow(N, 2.5);
    y->net_tx = 2e3 + 6e7 * pow(N * (0.4 + 0.6 * I), 2.5);
    y->disk_read = R > 0.02 ? 6e9 * pow(R, 2) : 0;
    y->disk_write = W > 0.02 ? 4e9 * pow(W, 2) : 0;
    for (int i = 0; i < N_GPUS; i++) {
        double g = i == 0 ? G : G * smooth(10, 14, u);
        g = clamp01(g * (0.94 + 0.06 * sin(u * 2.3 + i)));
        s->power[i] = 32 + 540 * g;
        s->load[i] = clamp01(g * 1.1);
        s->temp[i] = (int)(34 + 42 * g);
    }
    for (int i = 0; i < 32; i++)
        y->freq[i] = 600 + 5100 * clamp01(0.12 + 0.95 * y->cpu[i] + 0.1 * sin(u * 0.8 + i * 2.9));
    y->ctxt_s = 1.5e4 * pow(10, 2.2 * I);
    y->forks_s = 4 + 2800 * smooth(12, 14, u) * (1 - smooth(20, 23, u));      /* a big build */
    y->tcp_inuse = (int)(70 + 900 * N * N);
    y->cpu_watts = 38 + 185 * pow(I, 1.1);
    y->load1 = 32 * I;
    for (int i = 0; i < N_GPUS; i++)
        s->fan[i] = 0.3 + 0.65 * clamp01((s->power[i] - 32) / 540);
    s->tok_port[0] = G > 0.5 ? 700 * G : 0;
    s->tok_port[1] = G > 0.5 ? 640 * G : 0;
    s->tok_s = s->tok_port[0] + s->tok_port[1];
    s->running = s->tok_s > 0 ? 3 : 0;
}

/* ---------------------------------------------------------------- scene */

/*
 * The city is fixed geometry generated from a seed at startup. Silhouettes, dark
 * window grids, the road, the viaduct and the power plants are drawn once into
 * cached layers; each frame blits those and draws only the lit windows (grouped into
 * a handful of fills), glows, smoke, cars and trains. The sky and haze are rebuilt
 * only when the CPU temperature or load has visibly moved (4 times a second at
 * most), the water's reflections 10 times a second, the text 4 times a second.
 */
#define FPS_BUSY        20
#define FPS_IDLE        15
#define N_FRONT         16          /* physical cores 0..15 */
#define N_BACK          16          /* their SMT siblings 16..31 */
#define N_BLD           (N_FRONT + N_BACK)
#define FRONT_BASE      334         /* street level */
#define BACK_BASE       330
#define ROAD_Y0         334
#define ROAD_Y1         352
#define LANE_FAR        340.5       /* downloads: headlights heading left */
#define LANE_NEAR       347.5       /* uploads: taillights heading right */
#define RAIL_FAR        360         /* reads, heading left */
#define RAIL_NEAR       367         /* writes, heading right */
#define WATER_Y         370
#define MOON_X          312.0
#define MOON_Y          84.0
#define MOON_R          24.0
#define MAX_WIN         4096
#define MAX_PUFF        128
#define N_SPR           40
#define SPR_MIN         4.0
#define SPR_MAX         64.0
#define MAX_CARS        48
#define MAX_TRAINS      6
#define N_STARS         90
#define TEMP_COOL       40.0        /* sky: deep night blue at this Tctl ... */
#define TEMP_HOT        92.0        /* ... smoggy orange-red at this one */

typedef struct {
    int x, w, top, body_top, base;  /* body_top: below the setback tier, if any */
    int tier_w;
    int spire;
    int row, cpu;
    int w0, nw;                     /* its windows */
    cairo_surface_t *glow;
} bldg;

typedef struct { short x, y; unsigned char w, h, cool, bld; float base, ph, sp; } win;
typedef struct { double x, y, vx, vy, r0, age, life, a, lit; int alive; } puff;
typedef struct { double x, v, br; int lane, taxi, alive; } car;
typedef struct { double x, v; int n, track, alive; } train;
typedef struct { double x, y, r, ph, sp; } star;

/* The eased values the picture follows */
typedef struct {
    double cpu[MAX_CPUS], cpu_avg, temp, heat, ram;
    double act[N_GPUS], power[N_GPUS];
    double rx, tx, rd, wr;          /* 0..1 log-scaled */
    double tok;
    double clk[MAX_CPUS];           /* per-thread clock 0..1 (600..5700 MHz) */
    double ctxt, forks, tcp, cpuw;  /* 0..1 */
    double fan[N_GPUS];
} view_t;

static bldg    blds[N_BLD];
static win     wins[MAX_WIN];
static int     n_wins;
static puff    puffs[N_GPUS][MAX_PUFF];
static car     cars[MAX_CARS];
static train   trains[MAX_TRAINS];
static star    stars[N_STARS];
static double  puff_acc[N_GPUS][2], car_acc[2], train_gap[2], wind, beam_ang, taxi_acc, sig_phase, ring_acc;
#define N_RINGS 8
static double  rings[N_RINGS] = { -99, -99, -99, -99, -99, -99, -99, -99 };
static int     ring_next, mast_b = -1;
static const double SIGNAL_X[3] = { 130, 250, 370 };
#define MAST_H 34
static int     front_top[SIZE];     /* highest front-row pixel over each column */
static cairo_surface_t *sky_cache, *haze_cache, *moon_cache, *plant_cache, *back_cache, *front_cache;
static cairo_surface_t *refl_cache, *hud_cache;
static cairo_surface_t *smoke_spr[N_SPR], *lit_spr[N_GPUS][N_SPR], *plant_glow[N_GPUS], *red_spr, *taxi_spr, *sig_spr[3];
static cairo_surface_t *head_spr, *tail_spr, *train_spr[2];
static double  spr_r[N_SPR];
static char    hud_key[128];
static double  sky_heat = -1, sky_glow = -1, moon_f = -1, moon_heat = -1;

/* power plants: [0] GPU 0 on the left, [1] GPU 1 on the right */
static const double PLANT_X[N_GPUS]   = { 62, 418 };
static const double CHIMNEY_X[N_GPUS] = { 24, 456 };
#define TOWER_TOP   244.0
#define CHIMNEY_TOP 196.0
static const rgb PLANT_COL[N_GPUS] = { { 0.36, 0.72, 1.00 }, { 1.00, 0.55, 0.20 } };

static const rgb WARM  = { 1.00, 0.80, 0.46 };      /* lit windows */
static const rgb COOL  = { 0.66, 0.84, 1.00 };      /* the odd TV-blue window */
static const rgb RXC   = { 1.00, 0.95, 0.82 };      /* headlights */
static const rgb TXC   = { 1.00, 0.18, 0.12 };      /* taillights */
static const rgb READC = { 0.45, 0.95, 1.00 };      /* read trains */
static const rgb WRC   = { 0.62, 1.00, 0.42 };      /* write trains */
static const rgb TAXI  = { 1.00, 0.72, 0.18 };      /* taxis: new processes */

/* Sky colours at the top, middle and horizon, from cool night to smog */
static void sky_colors(double h, rgb *top, rgb *mid, rgb *hor)
{
    static const rgb T[4] = { { 0.012, 0.024, 0.078 }, { 0.024, 0.027, 0.11 }, { 0.06, 0.03, 0.10 }, { 0.11, 0.035, 0.045 } };
    static const rgb M[4] = { { 0.035, 0.075, 0.20 },  { 0.10, 0.09, 0.27 },   { 0.25, 0.10, 0.22 }, { 0.42, 0.13, 0.09 } };
    static const rgb H[4] = { { 0.11, 0.20, 0.38 },    { 0.32, 0.23, 0.46 },   { 0.68, 0.32, 0.30 }, { 0.93, 0.45, 0.16 } };
    double x = clamp01(h) * 3;
    int i = x >= 3 ? 2 : (int)x;
    double f = x - i;
    *top = lerp(T[i], T[i + 1], f);
    *mid = lerp(M[i], M[i + 1], f);
    *hor = lerp(H[i], H[i + 1], f);
}

static cairo_surface_t *radial_sprite(double r, rgb c, double a0, double a1, double a2)
{
    int w = (int)ceil(r * 2) + 2;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, w);
    cairo_t *cr = cairo_create(s);
    cairo_pattern_t *g = cairo_pattern_create_radial(w / 2.0, w / 2.0, 0, w / 2.0, w / 2.0, r);
    cairo_pattern_add_color_stop_rgba(g, 0.0, c.r, c.g, c.b, a0);
    cairo_pattern_add_color_stop_rgba(g, 0.35, c.r, c.g, c.b, a1);
    cairo_pattern_add_color_stop_rgba(g, 0.7, c.r, c.g, c.b, a2);
    cairo_pattern_add_color_stop_rgba(g, 1.0, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

/* A light moving sideways: a bright point with a long-exposure streak behind it */
static cairo_surface_t *streak_sprite(rgb c, double len, double core)
{
    int w = (int)len + 12, h = 12;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    double hx = w - 6, cy = h / 2.0;               /* the light is at the right end */
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, hx, 0);
    cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.75, c.r, c.g, c.b, 0.25);
    cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0.65);
    cairo_rectangle(cr, 0, cy - 1, hx, 2);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    g = cairo_pattern_create_radial(hx, cy, 0, hx, cy, 6);
    cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, 0.7);
    cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_arc(cr, hx, cy, 6, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    rgb w8 = lerp(c, (rgb){ 1, 1, 1 }, 0.6);
    cairo_set_source_rgba(cr, w8.r, w8.g, w8.b, 1);
    cairo_arc(cr, hx, cy, core, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_destroy(cr);
    return s;
}

/* A soft rectangle of light around a building: window light spilling into the night */
static cairo_surface_t *glow_sprite(int w, int h, double pad)
{
    int W = w + (int)(pad * 2), H = h + (int)(pad * 2);
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
    cairo_t *cr = cairo_create(s);
    for (int k = 0; k < 8; k++) {
        double inset = pad * k / 8.0;
        cairo_rectangle(cr, inset, inset + pad * 0.5, W - 2 * inset, H - 2 * inset - pad * 0.5);
        cairo_set_source_rgba(cr, WARM.r, WARM.g * 0.9, WARM.b * 0.8, 0.035 + 0.01 * k);
        cairo_fill(cr);
    }
    cairo_destroy(cr);
    return s;
}

/* Building geometry, window grids and the occlusion of the back row */
static void make_city(void)
{
    srand(20260925);
    const double slot = (462.0 - 18.0) / N_FRONT;
    for (int i = 0; i < N_FRONT; i++) {
        bldg *b = &blds[i];
        double cx = 18 + slot * (i + 0.5);
        double bell = exp(-pow((cx - 240) / 118.0, 2));
        int h = (int)(30 + 96 * bell + frand() * 26);
        if (i <= 1 || i >= N_FRONT - 2)
            h = 26 + (int)(frand() * 14);
        else if (i == 2 || i == N_FRONT - 3)
            h = 38 + (int)(frand() * 12);
        b->w = 21 + rand() % 6;
        b->x = (int)(cx - b->w / 2.0 + (frand() - 0.5) * 2);
        b->base = FRONT_BASE;
        b->top = b->base - h;
        b->tier_w = 0;
        b->body_top = b->top;
        if (h > 70 && frand() < 0.5) {
            b->tier_w = b->w - 6 - 2 * (rand() % 2);
            b->body_top = b->top + 10 + rand() % 14;
        }
        b->row = 0;
        b->cpu = i;
    }
    for (int x = 0; x < SIZE; x++)
        front_top[x] = FRONT_BASE;
    for (int i = 0; i < N_FRONT; i++) {
        bldg *b = &blds[i];
        for (int x = b->x; x < b->x + b->w; x++) {
            int tp = b->body_top;
            if (b->tier_w && x >= b->x + (b->w - b->tier_w) / 2 && x < b->x + (b->w + b->tier_w) / 2)
                tp = b->top;
            if (x >= 0 && x < SIZE && tp < front_top[x])
                front_top[x] = tp;
        }
    }
    const double bslot = (392.0 - 88.0) / N_BACK;
    for (int i = 0; i < N_BACK; i++) {
        bldg *b = &blds[N_FRONT + i];
        double cx = 88 + bslot * (i + 0.5);
        b->w = 14 + rand() % 4;
        b->x = (int)(cx - b->w / 2.0);
        int hi = BACK_BASE;
        for (int x = b->x; x < b->x + b->w; x++)
            if (front_top[x] < hi)
                hi = front_top[x];
        int h = BACK_BASE - hi + 24 + (int)(frand() * 46);
        h = h > 168 ? 168 : h < 70 ? 70 : h;
        b->base = BACK_BASE;
        b->top = b->base - h;
        b->tier_w = 0;
        b->body_top = b->top;
        if (frand() < 0.45) {
            b->tier_w = b->w - 4 - 2 * (rand() % 2);
            b->body_top = b->top + 8 + rand() % 14;
        }
        b->row = 1;
        b->cpu = N_FRONT + i;
    }
    /* spires with aviation lights on the three tallest of each row */
    for (int row = 0; row < 2; row++)
        for (int k = 0; k < 3; k++) {
            int best = -1;
            for (int i = row * N_FRONT; i < row * N_FRONT + 16; i++)
                if (!blds[i].spire && (best < 0 || blds[i].top < blds[best].top))
                    best = i;
            blds[best].spire = 8 + rand() % 12;
        }
    /* the radio mast goes on the tallest tower left of the moon */
    for (int i = N_FRONT; i < N_BLD; i++) {
        double cx = blds[i].x + blds[i].w / 2.0;
        if (cx > 140 && cx < 230 && (mast_b < 0 || blds[i].top < blds[mast_b].top))
            mast_b = i;
    }
    blds[mast_b].spire = 0;

    /* window grids, each window with its own place in the order the lights come on */
    n_wins = 0;
    for (int i = 0; i < N_BLD; i++) {
        bldg *b = &blds[i];
        int back = b->row == 1;
        int px = back ? 4 : 5, py = back ? 5 : 7, ww = back ? 2 : 3, wh = back ? 3 : 4;
        int cols = (b->w - 4 + (px - ww)) / px;
        int mx = (b->w - (cols * px - (px - ww))) / 2;
        b->w0 = n_wins;
        for (int y = b->body_top + 5; y + wh <= b->base - 3 && n_wins < MAX_WIN; y += py)
            for (int c = 0; c < cols && n_wins < MAX_WIN; c++) {
                int x = b->x + mx + c * px, vis = 1;
                if (back)
                    for (int xx = x; xx < x + ww; xx++)
                        if (y + wh > front_top[xx] - 1)
                            vis = 0;
                if (!vis)
                    continue;
                win *w = &wins[n_wins++];
                w->x = (short)x; w->y = (short)y; w->w = (unsigned char)ww; w->h = (unsigned char)wh;
                w->cool = frand() < 0.16;
                w->bld = (unsigned char)i;
            }
        b->nw = n_wins - b->w0;
        /* a shuffled rank per window: load L lights the lowest-ranked L of them */
        for (int k = 0; k < b->nw; k++)
            wins[b->w0 + k].base = (float)k;
        for (int k = b->nw - 1; k > 0; k--) {
            int j = rand() % (k + 1);
            float tmp = wins[b->w0 + k].base;
            wins[b->w0 + k].base = wins[b->w0 + j].base;
            wins[b->w0 + j].base = tmp;
        }
        for (int k = 0; k < b->nw; k++) {
            win *w = &wins[b->w0 + k];
            w->base = (float)(0.04 + 0.92 * (w->base + 0.5) / (b->nw ? b->nw : 1));
            w->ph = (float)(frand() * 6.3);
            w->sp = (float)(0.04 + frand() * 0.12);         /* people come and go, slowly */
        }
        b->glow = glow_sprite(b->w, b->base - b->top, back ? 7 : 10);
    }

    for (int i = 0; i < N_STARS; i++) {
        double a, r;
        star *st = &stars[i];
        do {
            a = frand() * 2 * M_PI;
            r = sqrt(frand()) * 228;
            st->x = 240 + cos(a) * r;
            st->y = 240 + sin(a) * r;
        } while (st->y > 250 || hypot(st->x - MOON_X, st->y - MOON_Y) < 40);
        st->r = 0.9 + frand() * frand() * 1.4;
        st->ph = frand() * 6.3;
        st->sp = 0.5 + frand() * 1.8;
    }
    srand((unsigned)time(NULL));
}

/* Hyperboloid cooling tower outline, centred at cx */
static void tower_path(cairo_t *cr, double cx)
{
    double top = TOWER_TOP, base = FRONT_BASE + 2;
    cairo_move_to(cr, cx - 36, base);
    cairo_curve_to(cr, cx - 26, base - 40, cx - 18, top + 40, cx - 21, top);
    cairo_line_to(cr, cx + 21, top);
    cairo_curve_to(cr, cx + 18, top + 40, cx + 26, base - 40, cx + 36, base);
    cairo_close_path(cr);
}

static void draw_building(cairo_t *cr, const bldg *b, rgb wall_top, rgb wall_bot, rgb rim, rgb dark_win)
{
    cairo_pattern_t *g = cairo_pattern_create_linear(0, b->top, 0, b->base);
    cairo_pattern_add_color_stop_rgb(g, 0, wall_top.r, wall_top.g, wall_top.b);
    cairo_pattern_add_color_stop_rgb(g, 1, wall_bot.r, wall_bot.g, wall_bot.b);
    cairo_set_source(cr, g);
    cairo_rectangle(cr, b->x, b->body_top, b->w, b->base - b->body_top + 2);
    if (b->tier_w)
        cairo_rectangle(cr, b->x + (b->w - b->tier_w) / 2, b->top, b->tier_w, b->body_top - b->top);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    if (b->spire) {
        double sx = b->x + b->w / 2.0;
        cairo_move_to(cr, sx - 1.5, b->top);
        cairo_line_to(cr, sx, b->top - b->spire);
        cairo_line_to(cr, sx + 1.5, b->top);
        set_rgb(cr, wall_top);
        cairo_fill(cr);
    }
    /* moonlight on the right-hand edges, a roof line */
    set_rgb(cr, rim);
    cairo_rectangle(cr, b->x + b->w - 1, b->body_top, 1, b->base - b->body_top);
    cairo_rectangle(cr, b->x, b->body_top, b->w, 1);
    if (b->tier_w) {
        cairo_rectangle(cr, b->x + (b->w + b->tier_w) / 2 - 1, b->top, 1, b->body_top - b->top);
        cairo_rectangle(cr, b->x + (b->w - b->tier_w) / 2, b->top, b->tier_w, 1);
    }
    cairo_fill(cr);
    set_rgb(cr, dark_win);
    for (int k = 0; k < b->nw; k++) {
        const win *w = &wins[b->w0 + k];
        cairo_rectangle(cr, w->x, w->y, w->w, w->h);
    }
    cairo_fill(cr);
}

static void build_caches(void)
{
    make_city();

    /* power plants */
    plant_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cairo_t *cr = cairo_create(plant_cache);
    for (int i = 0; i < N_GPUS; i++) {
        double cx = PLANT_X[i], chx = CHIMNEY_X[i];
        /* turbine hall */
        cairo_rectangle(cr, i == 0 ? cx + 18 : cx - 56, 296, 38, 44);
        cairo_set_source_rgb(cr, 0.07, 0.075, 0.12);
        cairo_fill(cr);
        /* chimney with faint bands */
        cairo_move_to(cr, chx - 4.5, FRONT_BASE);
        cairo_line_to(cr, chx - 3, CHIMNEY_TOP);
        cairo_line_to(cr, chx + 3, CHIMNEY_TOP);
        cairo_line_to(cr, chx + 4.5, FRONT_BASE);
        cairo_close_path(cr);
        cairo_set_source_rgb(cr, 0.085, 0.09, 0.14);
        cairo_fill(cr);
        for (int k = 0; k < 3; k++) {
            double y = CHIMNEY_TOP + 4 + k * 14;
            cairo_rectangle(cr, chx - 3.2 - k * 0.2, y, 6.4 + k * 0.4, 4);
        }
        cairo_set_source_rgb(cr, 0.16, 0.10, 0.12);
        cairo_fill(cr);
        /* cooling tower, shaded as a curved concrete shell */
        tower_path(cr, cx);
        cairo_pattern_t *g = cairo_pattern_create_linear(cx - 36, 0, cx + 36, 0);
        cairo_pattern_add_color_stop_rgb(g, 0.0, 0.07, 0.075, 0.12);
        cairo_pattern_add_color_stop_rgb(g, 0.6, 0.12, 0.13, 0.19);
        cairo_pattern_add_color_stop_rgb(g, 0.85, 0.15, 0.16, 0.23);
        cairo_pattern_add_color_stop_rgb(g, 1.0, 0.10, 0.11, 0.16);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
        cairo_save(cr);
        cairo_translate(cr, cx, TOWER_TOP);
        cairo_scale(cr, 21, 2.6);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgb(cr, 0.03, 0.03, 0.05);
        cairo_fill(cr);
        plant_glow[i] = radial_sprite(110, PLANT_COL[i], 0.55, 0.22, 0.06);
    }
    cairo_destroy(cr);

    /* back row: farther away, so hazier and bluer */
    back_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cr = cairo_create(back_cache);
    for (int i = N_FRONT; i < N_BLD; i++)
        draw_building(cr, &blds[i], (rgb){ 0.10, 0.11, 0.19 }, (rgb){ 0.07, 0.08, 0.14 },
                      (rgb){ 0.17, 0.19, 0.30 }, (rgb){ 0.12, 0.135, 0.22 });
    {   /* lattice radio mast */
        const bldg *b = &blds[mast_b];
        double mx = b->x + b->w / 2.0, y0 = b->top, y1 = b->top - MAST_H;
        cairo_set_line_width(cr, 1);
        cairo_set_source_rgb(cr, 0.20, 0.22, 0.34);
        cairo_move_to(cr, mx - 4, y0);
        cairo_line_to(cr, mx - 0.5, y1);
        cairo_move_to(cr, mx + 4, y0);
        cairo_line_to(cr, mx + 0.5, y1);
        for (int k = 0; k < 6; k++) {
            double ya = y0 - k * MAST_H / 6.0, yb = y0 - (k + 1) * MAST_H / 6.0;
            double wa = 4 - 3.5 * k / 6.0, wb = 4 - 3.5 * (k + 1) / 6.0;
            cairo_move_to(cr, mx - wa, ya);
            cairo_line_to(cr, mx + wb, yb);
        }
        cairo_stroke(cr);
        cairo_rectangle(cr, mx - 3, y0 - MAST_H * 0.55, 6, 2);
        cairo_fill(cr);
    }
    cairo_destroy(cr);

    /* front row, the waterfront road, street lamps and the rail viaduct */
    front_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cr = cairo_create(front_cache);
    for (int i = 0; i < N_FRONT; i++)
        draw_building(cr, &blds[i], (rgb){ 0.075, 0.08, 0.13 }, (rgb){ 0.035, 0.04, 0.07 },
                      (rgb){ 0.16, 0.18, 0.28 }, (rgb){ 0.085, 0.095, 0.15 });
    cairo_rectangle(cr, 0, ROAD_Y0, SIZE, ROAD_Y1 - ROAD_Y0);
    cairo_set_source_rgb(cr, 0.035, 0.037, 0.058);
    cairo_fill(cr);
    cairo_rectangle(cr, 0, ROAD_Y0, SIZE, 1.5);
    cairo_set_source_rgb(cr, 0.12, 0.12, 0.17);
    cairo_fill(cr);
    for (int x = 4; x < SIZE; x += 22)
        cairo_rectangle(cr, x, 343.5, 10, 1);
    cairo_set_source_rgba(cr, 0.75, 0.7, 0.55, 0.22);
    cairo_fill(cr);
    /* sodium street lamps: poles against the buildings, pools of light on the road */
    for (int x = 30; x < SIZE - 20; x += 40) {
        cairo_pattern_t *g = cairo_pattern_create_radial(x, 336, 0, x, 336, 26);
        cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.62, 0.25, 0.28);
        cairo_pattern_add_color_stop_rgba(g, 1, 1, 0.62, 0.25, 0);
        cairo_save(cr);
        cairo_translate(cr, x, 336);
        cairo_scale(cr, 1, 0.42);
        cairo_translate(cr, -x, -336);
        cairo_arc(cr, x, 336, 26, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
        cairo_rectangle(cr, x - 0.5, 322, 1, 13);
        cairo_rectangle(cr, x - 0.5, 322, 4, 1);
        cairo_set_source_rgb(cr, 0.05, 0.05, 0.07);
        cairo_fill(cr);
        g = cairo_pattern_create_radial(x + 3, 323.5, 0, x + 3, 323.5, 6);
        cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.85, 0.5, 1);
        cairo_pattern_add_color_stop_rgba(g, 0.25, 1, 0.66, 0.3, 0.5);
        cairo_pattern_add_color_stop_rgba(g, 1, 1, 0.6, 0.25, 0);
        cairo_arc(cr, x + 3, 323.5, 6, 0, 2 * M_PI);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    /* traffic signals */
    for (int k = 0; k < 3; k++) {
        double x = SIGNAL_X[k];
        cairo_rectangle(cr, x - 0.5, 318, 1, 17);
        cairo_rectangle(cr, x - 2, 309, 4, 10);
        cairo_set_source_rgb(cr, 0.04, 0.04, 0.06);
        cairo_fill(cr);
        for (int j = 0; j < 3; j++)
            cairo_rectangle(cr, x - 1, 310 + j * 3, 2, 2);
        cairo_set_source_rgb(cr, 0.13, 0.13, 0.16);
        cairo_fill(cr);
    }
    /* viaduct: deck, arches, two tracks */
    cairo_rectangle(cr, 0, ROAD_Y1, SIZE, WATER_Y - ROAD_Y1);
    cairo_set_source_rgb(cr, 0.03, 0.03, 0.05);
    cairo_fill(cr);
    for (int x = -10; x < SIZE; x += 30) {
        cairo_move_to(cr, x + 4, WATER_Y);
        cairo_curve_to(cr, x + 6, WATER_Y - 3.5, x + 24, WATER_Y - 3.5, x + 26, WATER_Y);
        cairo_close_path(cr);
    }
    cairo_set_source_rgb(cr, 0.06, 0.07, 0.11);
    cairo_fill(cr);
    cairo_rectangle(cr, 0, RAIL_FAR, SIZE, 1);
    cairo_rectangle(cr, 0, RAIL_NEAR, SIZE, 1);
    cairo_set_source_rgb(cr, 0.16, 0.17, 0.24);
    cairo_fill(cr);
    cairo_rectangle(cr, 0, ROAD_Y1, SIZE, 1);
    cairo_set_source_rgb(cr, 0.10, 0.10, 0.15);
    cairo_fill(cr);
    cairo_destroy(cr);

    for (int i = 0; i < N_SPR; i++) {
        spr_r[i] = SPR_MIN * pow(SPR_MAX / SPR_MIN, i / (N_SPR - 1.0));
        smoke_spr[i] = radial_sprite(spr_r[i], (rgb){ 0.72, 0.74, 0.82 }, 0.6, 0.42, 0.12);
        for (int k = 0; k < N_GPUS; k++)
            lit_spr[k][i] = radial_sprite(spr_r[i], lerp(PLANT_COL[k], (rgb){ 1, 1, 1 }, 0.35), 0.7, 0.45, 0.12);
    }
    red_spr = radial_sprite(7, (rgb){ 1, 0.12, 0.08 }, 0.9, 0.35, 0.08);
    taxi_spr = streak_sprite(TAXI, 26, 1.2);
    sig_spr[0] = radial_sprite(9, (rgb){ 0.2, 1.0, 0.45 }, 1.0, 0.4, 0.1);
    sig_spr[1] = radial_sprite(9, (rgb){ 1.0, 0.7, 0.1 }, 1.0, 0.4, 0.1);
    sig_spr[2] = radial_sprite(9, (rgb){ 1.0, 0.12, 0.08 }, 1.0, 0.4, 0.1);
    head_spr = streak_sprite(RXC, 34, 1.3);
    tail_spr = streak_sprite(TXC, 30, 1.1);
    train_spr[0] = radial_sprite(9, READC, 0.9, 0.3, 0.06);
    train_spr[1] = radial_sprite(9, WRC, 0.9, 0.3, 0.06);

    sky_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    haze_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    moon_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 120, 120);
    refl_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE - WATER_Y);
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
}

/* Sky, city glow and water, redrawn when the temperature or load has moved */
static void update_sky(const view_t *v)
{
    double heat = round(v->heat * 80) / 80, glow = round(v->cpuw * 40) / 40;
    if (fabs(heat - sky_heat) < 1e-6 && fabs(glow - sky_glow) < 1e-6)
        return;
    sky_heat = heat;
    sky_glow = glow;
    rgb top, mid, hor;
    sky_colors(heat, &top, &mid, &hor);

    cairo_t *cr = cairo_create(sky_cache);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, 0, FRONT_BASE);
    cairo_pattern_add_color_stop_rgb(g, 0.0, top.r, top.g, top.b);
    cairo_pattern_add_color_stop_rgb(g, 0.55, mid.r, mid.g, mid.b);
    cairo_pattern_add_color_stop_rgb(g, 1.0, hor.r, hor.g, hor.b);
    cairo_rectangle(cr, 0, 0, SIZE, FRONT_BASE);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    /* the city's own light pollution, brighter the more power the CPU package draws */
    rgb sod = lerp((rgb){ 1.0, 0.62, 0.32 }, hor, 0.3);
    cairo_save(cr);
    cairo_translate(cr, 240, FRONT_BASE);
    cairo_scale(cr, 1.6, 1);
    g = cairo_pattern_create_radial(0, 0, 0, 0, 0, 190);
    cairo_pattern_add_color_stop_rgba(g, 0, sod.r, sod.g, sod.b, 0.10 + 0.34 * glow);
    cairo_pattern_add_color_stop_rgba(g, 0.5, sod.r, sod.g, sod.b, 0.04 + 0.12 * glow);
    cairo_pattern_add_color_stop_rgba(g, 1, sod.r, sod.g, sod.b, 0);
    cairo_arc(cr, 0, 0, 190, M_PI, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    /* water: the horizon colour, darkened, fading into the deep */
    rgb w0 = lerp(hor, (rgb){ 0.02, 0.03, 0.06 }, 0.62), w1 = lerp(top, (rgb){ 0, 0, 0 }, 0.4);
    g = cairo_pattern_create_linear(0, FRONT_BASE, 0, SIZE);
    cairo_pattern_add_color_stop_rgb(g, 0, w0.r, w0.g, w0.b);
    cairo_pattern_add_color_stop_rgb(g, 1, w1.r, w1.g, w1.b);
    cairo_rectangle(cr, 0, FRONT_BASE, SIZE, SIZE - FRONT_BASE);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);

    /* haze between the two rows of buildings, for depth; smog when hot */
    cr = cairo_create(haze_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
    g = cairo_pattern_create_linear(0, 150, 0, FRONT_BASE);
    cairo_pattern_add_color_stop_rgba(g, 0, hor.r, hor.g, hor.b, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.6, hor.r, hor.g, hor.b, 0.10 + 0.14 * heat);
    cairo_pattern_add_color_stop_rgba(g, 1, hor.r, hor.g, hor.b, 0.30 + 0.25 * heat);
    cairo_rectangle(cr, 0, 150, SIZE, FRONT_BASE - 150);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
}

/* The moon: lit fraction = RAM in use, waxing from the right; it reddens in smog */
static void update_moon(const view_t *v)
{
    double f = round(v->ram * 200) / 200, heat = round(v->heat * 20) / 20;
    if (fabs(f - moon_f) < 1e-6 && fabs(heat - moon_heat) < 1e-6)
        return;
    moon_f = f;
    moon_heat = heat;
    rgb c = lerp((rgb){ 0.97, 0.95, 0.86 }, (rgb){ 1.0, 0.66, 0.42 }, heat * 0.8);
    const double r = MOON_R, cx = 60, cy = 60;

    cairo_t *cr = cairo_create(moon_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_pattern_t *g = cairo_pattern_create_radial(cx, cy, r * 0.8, cx, cy, 60);
    cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, 0.10 + 0.22 * f);
    cairo_pattern_add_color_stop_rgba(g, 0.4, c.r, c.g, c.b, 0.03 + 0.07 * f);
    cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);

    /* earthshine on the dark part */
    cairo_arc(cr, cx, cy, r, 0, 2 * M_PI);
    cairo_set_source_rgba(cr, 0.16, 0.18, 0.26, 0.9);
    cairo_fill(cr);

    /* lit part: right half-disc, then back up along the terminator ellipse */
    double rx = fmax(0.02, fabs(1 - 2 * f));
    cairo_new_path(cr);
    cairo_arc(cr, cx, cy, r, -M_PI / 2, M_PI / 2);
    cairo_save(cr);
    cairo_translate(cr, cx, cy);
    cairo_scale(cr, rx, 1);
    if (f < 0.5)
        cairo_arc_negative(cr, 0, 0, r, M_PI / 2, -M_PI / 2);
    else
        cairo_arc(cr, 0, 0, r, M_PI / 2, 3 * M_PI / 2);
    cairo_restore(cr);
    cairo_close_path(cr);
    g = cairo_pattern_create_radial(cx - r * 0.3, cy - r * 0.3, 0, cx, cy, r * 1.1);
    cairo_pattern_add_color_stop_rgb(g, 0, fmin(1, c.r * 1.05), fmin(1, c.g * 1.05), fmin(1, c.b * 1.05));
    cairo_pattern_add_color_stop_rgb(g, 1, c.r * 0.82, c.g * 0.80, c.b * 0.78);
    cairo_set_source(cr, g);
    cairo_fill_preserve(cr);
    cairo_pattern_destroy(g);
    /* maria */
    cairo_clip(cr);
    static const double mar[5][3] = { { -6, -8, 7 }, { 7, -3, 5 }, { 2, 8, 6 }, { -9, 5, 4 }, { 10, 10, 3 } };
    for (int k = 0; k < 5; k++) {
        cairo_arc(cr, cx + mar[k][0], cy + mar[k][1], mar[k][2], 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 0.45, 0.42, 0.40, 0.18);
        cairo_fill(cr);
    }
    cairo_destroy(cr);
}

/* ---------------------------------------------------------------- simulation */

static void spawn_puff(int p, int chimney, double act, double fan)
{
    for (int i = 0; i < MAX_PUFF; i++) {
        puff *f = &puffs[p][i];
        if (f->alive)
            continue;
        f->alive = 1;
        f->x = (chimney ? CHIMNEY_X[p] : PLANT_X[p]) + (frand() - 0.5) * (chimney ? 3 : 26);
        f->y = chimney ? CHIMNEY_TOP - 2 : TOWER_TOP - 3;
        f->vx = (frand() - 0.5) * 6;
        f->vy = -(12 + 20 * act) * (0.8 + frand() * 0.4) * (chimney ? 0.8 : 1) * (0.65 + 0.7 * fan);
        f->r0 = chimney ? 3 : 12 + frand() * 5;
        f->age = 0;
        f->life = (chimney ? 5 : 6) + frand() * 3 + 2 * act;
        f->a = (chimney ? 0.35 : 0.75) * (0.3 + 0.7 * act);
        f->lit = chimney ? 0.3 : 1;
        return;
    }
}

static void spawn_car(int lane, double busy, int taxi)
{
    for (int i = 0; i < MAX_CARS; i++) {
        car *c = &cars[i];
        if (c->alive)
            continue;
        c->alive = 1;
        c->lane = lane;
        c->taxi = taxi;
        c->v = (70 + 110 * busy) * (0.85 + frand() * 0.3) * (lane ? 1 : -1);
        c->x = lane ? -40 : SIZE + 40;
        c->br = 0.7 + frand() * 0.3;
        return;
    }
}

static int simulate(const view_t *v, double dt, double t)
{
    int active = 0;
    wind = 5 * sin(t * 0.05) + 2;

    /* steam and smoke from the plants */
    for (int p = 0; p < N_GPUS; p++) {
        double a = v->act[p];
        puff_acc[p][0] += (1.0 + 5.5 * a) * dt;
        puff_acc[p][1] += (0.6 + 2.5 * a) * dt;
        for (int k = 0; k < 2; k++)
            while (puff_acc[p][k] >= 1) {
                spawn_puff(p, k, a, v->fan[p]);
                puff_acc[p][k] -= 1;
            }
        for (int i = 0; i < MAX_PUFF; i++) {
            puff *f = &puffs[p][i];
            if (!f->alive)
                continue;
            f->age += dt;
            f->vy *= 1 - 0.12 * dt;
            f->x += (f->vx + wind * clamp01(f->age / 3)) * dt;
            f->y += f->vy * dt;
            if (f->age > f->life || f->y < 10)
                f->alive = 0;
        }
    }

    /* cars: downloads are headlights heading left, uploads taillights heading right */
    double nets[2] = { v->rx, v->tx };
    for (int lane = 0; lane < 2; lane++) {
        double n = nets[lane];
        car_acc[lane] += (n > 0.02 ? 0.25 + 7.5 * n * n : 0) * dt;
        while (car_acc[lane] >= 1) {
            spawn_car(lane, n, 0);
            car_acc[lane] -= 1 - frand() * 0.4;
        }
    }
    /* taxis: every burst of new processes sends more of them into both lanes */
    taxi_acc += (v->forks > 0.08 ? 5.0 * pow(v->forks, 1.5) : 0) * dt;
    while (taxi_acc >= 1) {
        spawn_car(rand() & 1, 0.5, 1);
        taxi_acc -= 1 - frand() * 0.4;
    }
    /* traffic signals cycle faster with more context switches */
    sig_phase += dt / (7.0 * pow(0.12, v->ctxt));
    /* the radio mast sends out a ring per burst of open connections */
    ring_acc += (0.25 + 2.2 * v->tcp * v->tcp) * dt;
    if (ring_acc >= 1) {
        ring_acc -= 1;
        rings[ring_next] = t;
        ring_next = (ring_next + 1) % N_RINGS;
    }
    for (int i = 0; i < MAX_CARS; i++) {
        car *c = &cars[i];
        if (!c->alive)
            continue;
        active++;
        c->x += c->v * dt;
        if (c->x < -60 || c->x > SIZE + 60)
            c->alive = 0;
    }

    /* trains: reads heading left on the far track, writes heading right on the near one */
    double io[2] = { v->rd, v->wr };
    for (int k = 0; k < 2; k++) {
        train_gap[k] -= dt;
        if (io[k] > 0.04 && train_gap[k] <= 0) {
            for (int i = 0; i < MAX_TRAINS; i++) {
                train *tr = &trains[i];
                if (tr->alive)
                    continue;
                tr->alive = 1;
                tr->track = k;
                tr->n = 2 + (int)(io[k] * 6.99);
                tr->v = (80 + 240 * io[k]) * (k ? 1 : -1);
                double len = tr->n * 24;
                tr->x = k ? -len - 10 : SIZE + 10;
                /* the next one follows closely when the drive is busy */
                train_gap[k] = (len + 30 + 260 * (1 - io[k])) / fabs(tr->v);
                break;
            }
        }
    }
    for (int i = 0; i < MAX_TRAINS; i++) {
        train *tr = &trains[i];
        if (!tr->alive)
            continue;
        active++;
        tr->x += tr->v * dt;
        double len = tr->n * 24;
        if ((tr->v < 0 && tr->x + len < -20) || (tr->v > 0 && tr->x > SIZE + 20))
            tr->alive = 0;
    }
    beam_ang += dt * (0.25 + 0.5 * clamp01(v->tok / 1200));
    return active;
}

/* ---------------------------------------------------------------- render */

static void blit(cairo_t *cr, cairo_surface_t *spr, double x, double y, double alpha)
{
    double w = cairo_image_surface_get_width(spr), h = cairo_image_surface_get_height(spr);
    cairo_set_source_surface(cr, spr, round(x - w / 2), round(y - h / 2));
    cairo_paint_with_alpha(cr, alpha);         /* EXTEND_NONE: only the sprite's area is touched */
}

/* Paint a cached full-frame layer, touching only the rows it covers */
static void paint_band(cairo_t *cr, cairo_surface_t *s, double y0, double y1, double alpha)
{
    cairo_save(cr);
    cairo_rectangle(cr, 0, y0, SIZE, y1 - y0);
    cairo_clip(cr);
    cairo_set_source_surface(cr, s, 0, 0);
    cairo_paint_with_alpha(cr, alpha);
    cairo_restore(cr);
}

static int spr_index(double r)
{
    int i = (int)lround(log(r / SPR_MIN) / log(SPR_MAX / SPR_MIN) * (N_SPR - 1));
    return i < 0 ? 0 : i >= N_SPR ? N_SPR - 1 : i;
}

/* Lit windows of one row, batched into a few fills by colour and brightness */
#define N_LEVELS 5
static void draw_windows(cairo_t *cr, const view_t *v, int row, double t)
{
    static int idx[2][N_LEVELS][MAX_WIN];
    int cnt[2][N_LEVELS] = { { 0 } };
    double dim = row ? 0.72 : 1.0;

    for (int i = row * N_FRONT; i < row * N_FRONT + (row ? N_BACK : N_FRONT); i++) {
        const bldg *b = &blds[i];
        double load = v->cpu[b->cpu];
        if (load < 0.005)
            continue;
        for (int k = 0; k < b->nw; k++) {
            const win *w = &wins[b->w0 + k];
            double s = w->base + 0.035 * sin(t * w->sp + w->ph);
            double br = clamp01((load - s) / 0.035);
            if (br <= 0.1)
                continue;
            int lv = (int)ceil(br * N_LEVELS) - 1;
            idx[w->cool][lv][cnt[w->cool][lv]++] = b->w0 + k;
        }
    }
    for (int c = 0; c < 2; c++)
        for (int lv = 0; lv < N_LEVELS; lv++) {
            if (!cnt[c][lv])
                continue;
            for (int k = 0; k < cnt[c][lv]; k++) {
                const win *w = &wins[idx[c][lv][k]];
                cairo_rectangle(cr, w->x, w->y, w->w, w->h);
            }
            rgb col = c ? COOL : WARM;
            cairo_set_source_rgba(cr, col.r, col.g, col.b, dim * (lv + 1) / (double)N_LEVELS);
            cairo_fill(cr);
        }
}

/* Window light spilling around busy buildings */
static void draw_glows(cairo_t *cr, const view_t *v, int row)
{
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = row * N_FRONT; i < row * N_FRONT + 16; i++) {
        const bldg *b = &blds[i];
        double load = v->cpu[b->cpu], a = pow(smooth(0.35, 1.0, load), 1.6) * (row ? 0.12 : 0.35);
        if (a > 0.01) {
            int W = cairo_image_surface_get_width(b->glow), H = cairo_image_surface_get_height(b->glow);
            cairo_set_source_surface(cr, b->glow, b->x + b->w / 2 - W / 2, b->top - (H - (b->base - b->top)) / 2);
            cairo_paint_with_alpha(cr, a);
        }
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

/* Neon along each roof, coloured by that thread's clock: deep blue when parked, teal, hot pink at full boost */
static rgb neon_color(double c)
{
    static const rgb lo = { 0.25, 0.35, 1.00 }, mid = { 0.15, 1.00, 0.85 }, hi = { 1.00, 0.22, 0.72 };
    return c < 0.5 ? lerp(lo, mid, c / 0.5) : lerp(mid, hi, (c - 0.5) / 0.5);
}

static void draw_neon(cairo_t *cr, const view_t *v, int row)
{
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = row * N_FRONT; i < row * N_FRONT + 16; i++) {
        const bldg *b = &blds[i];
        double c = v->clk[b->cpu], a = (0.35 + 0.65 * c) * (row ? 0.8 : 1);
        int rw = b->tier_w ? b->tier_w : b->w, rx = b->x + (b->w - rw) / 2;
        rgb n = neon_color(c);
        cairo_set_source_rgba(cr, n.r, n.g, n.b, a);
        cairo_rectangle(cr, rx, b->top, rw, 2);
        cairo_fill(cr);
        cairo_set_source_rgba(cr, n.r, n.g, n.b, a * 0.22);
        cairo_rectangle(cr, rx - 1, b->top - 3, rw + 2, 3);
        cairo_rectangle(cr, rx, b->top + 2, rw, 3);
        cairo_fill(cr);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

/* Radio waves from the mast, one per burst of open TCP connections */
static void draw_mast(cairo_t *cr, double t)
{
    const bldg *b = &blds[mast_b];
    double mx = b->x + b->w / 2.0, my = b->top - MAST_H;
    cairo_set_line_width(cr, 1.6);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int k = 0; k < N_RINGS; k++) {
        double a = t - rings[k];
        if (a < 0 || a > 2.6)
            continue;
        double r = 4 + a * 26, al = 0.6 * pow(1 - a / 2.6, 1.4);
        cairo_new_path(cr);
        cairo_arc(cr, mx, my, r, -M_PI + 0.55, -0.55);
        cairo_set_source_rgba(cr, 0.55, 1.0, 0.9, al);
        cairo_stroke(cr);
    }
    double ph = fmod(t * 1.1, 1.0), a = ph < 0.12 ? 1 : 0.25;          /* white strobe on the tip */
    cairo_rectangle(cr, mx - 1, my - 1, 2, 2);
    cairo_set_source_rgba(cr, 1, 1, 1, a);
    cairo_fill(cr);
}

static void draw_signals(cairo_t *cr)
{
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int k = 0; k < 3; k++) {
        double ph = fmod(sig_phase + k * 0.37, 1.0);
        int st = ph < 0.5 ? 0 : ph < 0.62 ? 1 : 2;        /* green, amber, red */
        double x = SIGNAL_X[k], y = 310 + (2 - st) * 3 + 1;
        blit(cr, sig_spr[st], x, y, 0.9);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

static void draw_aviation(cairo_t *cr, int row, double t)
{
    for (int i = row * N_FRONT; i < row * N_FRONT + 16; i++) {
        const bldg *b = &blds[i];
        if (!b->spire)
            continue;
        double ph = fmod(t * 0.7 + i * 0.37, 1.0), a = ph < 0.35 ? sin(ph / 0.35 * M_PI) : 0;
        double x = b->x + b->w / 2.0, y = b->top - b->spire;
        cairo_rectangle(cr, x - 0.5, y - 0.5, 1.5, 1.5);
        cairo_set_source_rgba(cr, 0.55, 0.1, 0.08, 0.9);
        cairo_fill(cr);
        if (a > 0.01)
            blit(cr, red_spr, x + 0.25, y + 0.25, a * (row ? 0.7 : 1));
    }
}

static void draw_plants(cairo_t *cr, const view_t *v, double t)
{
    rgb top, mid, hor;
    sky_colors(v->heat, &top, &mid, &hor);
    (void)top;
    (void)mid;

    /* glow of the plant's lights in the air, behind everything */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int p = 0; p < N_GPUS; p++)
        blit(cr, plant_glow[p], PLANT_X[p], TOWER_TOP + 10, 0.10 + 0.75 * v->act[p]);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* steam: grey against the sky, lit from below near the tower mouth */
    for (int p = 0; p < N_GPUS; p++)
        for (int i = 0; i < MAX_PUFF; i++) {
            const puff *f = &puffs[p][i];
            if (!f->alive)
                continue;
            double r = f->r0 + f->age * (f->lit > 0.5 ? 5.5 : 2.2);
            double fade = clamp01(f->age / 0.4) * pow(1 - f->age / f->life, 1.5);
            double a = f->a * fade;
            double lit = f->lit * clamp01(1 - (TOWER_TOP - f->y) / 110) * (0.2 + 0.8 * v->act[p]);
            double edge = clamp01((236 - hypot(f->x - 240, f->y - 240)) / 20);
            a *= edge;
            if (a < 0.01)
                continue;
            int k = spr_index(r);
            blit(cr, smoke_spr[k], f->x, f->y, a * (1 - lit * 0.6));
            if (lit > 0.03)
                blit(cr, lit_spr[p][k], f->x, f->y, a * lit);
        }

    paint_band(cr, plant_cache, CHIMNEY_TOP - 2, FRONT_BASE, 1);

    /* the glowing mouth of each tower, lights on the chimneys and the hall */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int p = 0; p < N_GPUS; p++) {
        double a = v->act[p];
        cairo_save(cr);
        cairo_translate(cr, PLANT_X[p], TOWER_TOP);
        cairo_scale(cr, 21, 2.6);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, PLANT_COL[p].r, PLANT_COL[p].g, PLANT_COL[p].b, 0.15 + 0.7 * a);
        cairo_fill(cr);
        double hx = p == 0 ? PLANT_X[p] + 22 : PLANT_X[p] - 52;
        for (int k = 0; k < 5; k++)
            cairo_rectangle(cr, hx + k * 6.5, 300, 3, 3);
        cairo_set_source_rgba(cr, PLANT_COL[p].r, PLANT_COL[p].g, PLANT_COL[p].b, 0.3 + 0.6 * a);
        cairo_fill(cr);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    for (int p = 0; p < N_GPUS; p++)
        for (int k = 0; k < 2; k++) {
            double ph = fmod(t * 0.5 + p * 0.5 + k * 0.1, 1.0), a = ph < 0.3 ? sin(ph / 0.3 * M_PI) : 0;
            double y = CHIMNEY_TOP + 2 + k * 40;
            cairo_rectangle(cr, CHIMNEY_X[p] - 0.75, y - 0.75, 1.5, 1.5);
            cairo_set_source_rgba(cr, 0.55, 0.1, 0.08, 0.9);
            cairo_fill(cr);
            if (a > 0.01)
                blit(cr, red_spr, CHIMNEY_X[p], y, a);
        }
}

static void draw_stars(cairo_t *cr, const view_t *v, double t)
{
    double vis = 1 - 0.85 * smooth(0.1, 0.9, v->heat);
    for (int i = 0; i < N_STARS; i++) {
        const star *st = &stars[i];
        double a = 0.5 + 0.5 * sin(t * st->sp + st->ph);
        a = (0.25 + 0.75 * a * a) * vis * clamp01((260 - st->y) / 90);
        if (a < 0.03)
            continue;
        cairo_rectangle(cr, st->x - st->r / 2, st->y - st->r / 2, st->r, st->r);
        cairo_set_source_rgba(cr, 0.88, 0.92, 1, a);
        cairo_fill(cr);
    }
}

/* vLLM bonus: two searchlights sweep the sky from downtown while tokens flow */
static void draw_beams(cairo_t *cr, const view_t *v)
{
    double a = clamp01(v->tok / 150);
    if (a < 0.02)
        return;
    int src[2] = { 6, 9 };
    for (int k = 0; k < 2; k++) {
        const bldg *b = &blds[src[k]];
        double x0 = b->x + b->w / 2.0, y0 = b->top;
        double ang = -M_PI / 2 + 0.55 * sin(beam_ang * (k ? 1.13 : 1) + k * 2.2);
        double L = 300, sp = 0.05;
        double x1 = x0 + cos(ang - sp) * L, y1 = y0 + sin(ang - sp) * L;
        double x2 = x0 + cos(ang + sp) * L, y2 = y0 + sin(ang + sp) * L;
        cairo_pattern_t *g = cairo_pattern_create_linear(x0, y0, x0 + cos(ang) * L, y0 + sin(ang) * L);
        cairo_pattern_add_color_stop_rgba(g, 0, 0.85, 0.9, 1, 0.16 * a);
        cairo_pattern_add_color_stop_rgba(g, 1, 0.85, 0.9, 1, 0);
        cairo_move_to(cr, x0 - 1.5, y0);
        cairo_line_to(cr, x1, y1);
        cairo_line_to(cr, x2, y2);
        cairo_line_to(cr, x0 + 1.5, y0);
        cairo_close_path(cr);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
}

static void draw_traffic(cairo_t *cr)
{
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = 0; i < MAX_CARS; i++) {
        const car *c = &cars[i];
        if (!c->alive)
            continue;
        cairo_surface_t *s = c->taxi ? taxi_spr : c->lane ? tail_spr : head_spr;
        double w = cairo_image_surface_get_width(s), y = c->lane ? LANE_NEAR : LANE_FAR;
        cairo_save(cr);
        if (c->lane) {
            cairo_set_source_surface(cr, s, round(c->x - w + 6), y - 6);
        } else {                                    /* mirrored: heading left, streak behind to the right */
            cairo_translate(cr, round(c->x), 0);
            cairo_scale(cr, -1, 1);
            cairo_set_source_surface(cr, s, -w + 6, y - 6);
        }
        cairo_paint_with_alpha(cr, c->br);
        cairo_restore(cr);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

static void draw_trains(cairo_t *cr)
{
    for (int i = 0; i < MAX_TRAINS; i++) {
        const train *tr = &trains[i];
        if (!tr->alive)
            continue;
        double yb = tr->track ? RAIL_NEAR : RAIL_FAR, yt = yb - 7;
        rgb c = tr->track ? WRC : READC;
        double x0 = round(tr->x);
        /* carriages */
        for (int k = 0; k < tr->n; k++) {
            double x = x0 + k * 24;
            cairo_rectangle(cr, x, yt, 22, 7);
        }
        cairo_set_source_rgb(cr, 0.08, 0.085, 0.12);
        cairo_fill(cr);
        for (int k = 0; k < tr->n; k++) {
            double x = x0 + k * 24;
            for (int j = 0; j < 5; j++)
                cairo_rectangle(cr, x + 2 + j * 4, yt + 2, 3, 2);
        }
        cairo_set_source_rgba(cr, c.r, c.g, c.b, 0.95);
        cairo_fill(cr);
        /* its light in the water below */
        for (int k = 0; k < tr->n; k++)
            cairo_rectangle(cr, x0 + k * 24 + 2, WATER_Y + 2 + (tr->track ? 1 : 0), 19, 1);
        cairo_set_source_rgba(cr, c.r, c.g, c.b, 0.22);
        cairo_fill(cr);
        /* the headlight at the front */
        double hx = tr->v < 0 ? x0 - 1 : x0 + tr->n * 24 - 1;
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        blit(cr, train_spr[tr->track], hx, yt + 4, 0.9);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    }
}

/* Broken, rippling reflections of the city in the water, 10 times a second */
static void update_reflections(const view_t *v, double t)
{
    cairo_t *cr = cairo_create(refl_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_translate(cr, 0, -WATER_Y);
    for (int i = 0; i < N_BLD; i++) {
        const bldg *b = &blds[i];
        double load = v->cpu[b->cpu], cx = b->x + b->w / 2.0;
        double a0 = (0.08 + 0.55 * load) * (b->row ? 0.6 : 1);
        if (load < 0.03)
            continue;
        for (int k = 0; k < 20; k++) {
            double y = WATER_Y + 3 + k * 4.2 + (b->row ? 2 : 0);
            double depth = k / 20.0;
            double wob = sin(t * 1.9 + k * 1.3 + i * 0.7) * (1 + 3 * depth);
            double w = b->w * (0.35 + 0.45 * (0.5 + 0.5 * sin(t * 2.3 + k * 2.1 + i))) * (b->row ? 0.7 : 1);
            double a = a0 * (1 - depth) * (0.4 + 0.6 * (0.5 + 0.5 * sin(t * 3.1 + k * 0.9 + i * 1.7)));
            cairo_rectangle(cr, round(cx + wob - w / 2), round(y), round(w), 1.5);
            cairo_set_source_rgba(cr, WARM.r, WARM.g * 0.92, WARM.b * 0.85, a);
            cairo_fill(cr);
        }
    }
    /* the plants' glow and the street lamps */
    for (int p = 0; p < N_GPUS; p++) {
        double a = 0.05 + 0.4 * v->act[p];
        for (int k = 0; k < 16; k++) {
            double y = WATER_Y + 3 + k * 5, depth = k / 16.0;
            double w = 26 * (0.5 + 0.5 * sin(t * 2 + k * 1.7 + p)) * (1 - 0.4 * depth);
            cairo_rectangle(cr, round(PLANT_X[p] - w / 2 + sin(t * 1.5 + k) * 3), y, round(w), 1.5);
            cairo_set_source_rgba(cr, PLANT_COL[p].r, PLANT_COL[p].g, PLANT_COL[p].b, a * (1 - depth));
            cairo_fill(cr);
        }
    }
    for (int x = 30; x < SIZE - 20; x += 40)
        for (int k = 0; k < 6; k++) {
            double w = 6 * (0.5 + 0.5 * sin(t * 2.7 + k * 2.3 + x));
            cairo_rectangle(cr, round(x + 3 - w / 2 + sin(t * 1.7 + k + x) * 1.5), WATER_Y + 4 + k * 5, round(w), 1);
        }
    cairo_set_source_rgba(cr, 1, 0.66, 0.3, 0.25);
    cairo_fill(cr);
    cairo_destroy(cr);
}

static void soft_text(cairo_t *cr, double x, double y, double size, rgb c, const char *s, double maxw)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    if (maxw > 0 && ext.width > maxw) {                /* fit the chord */
        size *= maxw / ext.width;
        cairo_set_font_size(cr, size);
        cairo_text_extents(cr, s, &ext);
    }
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    cairo_text_path(cr, s);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, size * 0.22);
    cairo_set_source_rgba(cr, 0.01, 0.015, 0.05, 0.8);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
}

/* A small label and a big number side by side, sharing a baseline, centred on (x, y) */
static void text_pair(cairo_t *cr, double x, double y, const char *a, double sa, rgb ca,
                      const char *b, double sb, rgb cb, double maxw)
{
    cairo_text_extents_t ea, eb;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, sb);
    cairo_text_extents(cr, b, &eb);
    cairo_set_font_size(cr, sa);
    cairo_text_extents(cr, a, &ea);
    double gap = sa * 0.4, w = ea.x_advance + gap + eb.width;
    if (w > maxw) {
        double k = maxw / w;
        sa *= k, sb *= k, gap *= k, w = maxw;
        cairo_set_font_size(cr, sb);
        cairo_text_extents(cr, b, &eb);
        cairo_set_font_size(cr, sa);
        cairo_text_extents(cr, a, &ea);
    }
    double base = y - eb.height / 2 - eb.y_bearing, x0 = x - w / 2;
    const char *str[2] = { a, b };
    double sz[2] = { sa, sb }, xs[2] = { x0 - ea.x_bearing, x0 + ea.x_advance + gap - eb.x_bearing };
    rgb col[2] = { ca, cb };
    for (int i = 0; i < 2; i++) {
        cairo_set_font_size(cr, sz[i]);
        cairo_move_to(cr, xs[i], base);
        cairo_text_path(cr, str[i]);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
        cairo_set_line_width(cr, sz[i] * 0.22);
        cairo_set_source_rgba(cr, 0.01, 0.015, 0.05, 0.8);
        cairo_stroke_preserve(cr);
        set_rgb(cr, col[i]);
        cairo_fill(cr);
    }
}

static double chord(double y, double margin) { double d = y - 240; return 2 * sqrt(fmax(0, 240.0 * 240.0 - d * d)) - 2 * margin; }

/* Shown numbers only move when the value has really moved (no digits flipping) */
static double hyst(double v, double *shown, double band)
{
    if (isnan(*shown) || fabs(v - *shown) > band)
        *shown = round(v);
    return *shown;
}

static void update_hud(const view_t *v, double t)
{
    static double next, cpu = NAN, temp = NAN, ram = NAN, watts = NAN;
    char key[128], txt[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        hyst(v->cpu_avg * 100, &cpu, 0.8);
        hyst(v->temp, &temp, 0.8);
        hyst(v->ram * 100, &ram, 0.8);
        hyst(v->power[0] + v->power[1], &watts, 4);
    }
    int heat_i = (int)(v->heat * 10);
    snprintf(key, sizeof(key), "%.0f|%.0f|%.0f|%.0f|%d", cpu, temp, ram, watts, heat_i);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);

    rgb top, mid, hor;
    sky_colors(v->heat, &top, &mid, &hor);
    (void)top;
    (void)mid;
    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    snprintf(txt, sizeof(txt), "RAM %.0f%%", ram);
    soft_text(cr, MOON_X, MOON_Y + MOON_R + 20, 22, (rgb){ 0.92, 0.92, 0.88 }, txt, 0);

    snprintf(txt, sizeof(txt), "%.0f%%", cpu);
    text_pair(cr, 240, 404, "CPU", 22, (rgb){ 0.62, 0.65, 0.76 }, txt, 38, lerp(WARM, (rgb){ 1, 1, 1 }, 0.35), chord(404, 30));
    snprintf(txt, sizeof(txt), "%.0f\xC2\xB0" "C", temp);
    soft_text(cr, 240 - 62, 442, 24, lerp(hor, (rgb){ 1, 1, 1 }, 0.45), txt, 100);
    snprintf(txt, sizeof(txt), "%.0f W", watts);
    soft_text(cr, 240 + 62, 442, 24, lerp(lerp(PLANT_COL[0], PLANT_COL[1], 0.5), (rgb){ 1, 1, 1 }, 0.3), txt, 110);
    cairo_destroy(cr);
}

static void render(cairo_t *cr, const view_t *v, double t)
{
    static double next_refl = -1;
    update_sky(v);
    update_moon(v);

    cairo_set_source_surface(cr, sky_cache, 0, 0);
    cairo_paint(cr);
    draw_stars(cr, v, t);
    cairo_set_source_surface(cr, moon_cache, MOON_X - 60, MOON_Y - 60);
    cairo_paint(cr);
    draw_beams(cr, v);
    draw_plants(cr, v, t);

    paint_band(cr, back_cache, 140, BACK_BASE + 2, 1);
    draw_glows(cr, v, 1);
    draw_windows(cr, v, 1, t);
    draw_neon(cr, v, 1);
    draw_aviation(cr, 1, t);
    draw_mast(cr, t);
    paint_band(cr, haze_cache, 150, FRONT_BASE, 1);

    paint_band(cr, front_cache, 190, WATER_Y, 1);
    draw_glows(cr, v, 0);
    draw_windows(cr, v, 0, t);
    draw_neon(cr, v, 0);
    draw_aviation(cr, 0, t);
    draw_signals(cr);
    draw_traffic(cr);
    draw_trains(cr);

    if (t >= next_refl || t < next_refl - 1) {
        next_refl = t + 0.1;
        update_reflections(v, t);
    }
    cairo_set_source_surface(cr, refl_cache, 0, WATER_Y);
    cairo_paint(cr);

    update_hud(v, t);
    paint_band(cr, hud_cache, MOON_Y + MOON_R + 4, MOON_Y + MOON_R + 36, 1);
    paint_band(cr, hud_cache, 380, 460, 1);
}

/* Ease the picture toward the latest readings */
static void ease_view(view_t *v, const sys_stats *y, const stats *s, double dt)
{
    double k = fmin(1, dt * 3), ks = fmin(1, dt * 0.8), kn = fmin(1, dt * 2);
    for (int i = 0; i < MAX_CPUS; i++)
        v->cpu[i] += (y->cpu[i] - v->cpu[i]) * k;
    v->cpu_avg += (y->cpu_avg - v->cpu_avg) * k;
    if (y->cpu_temp > 0)
        v->temp += (y->cpu_temp - v->temp) * ks;
    v->heat = clamp01((v->temp - TEMP_COOL) / (TEMP_HOT - TEMP_COOL));
    v->ram += (y->ram_frac - v->ram) * kn;
    for (int i = 0; i < N_GPUS; i++) {
        double pw = clamp01((s->power[i] - GPU_IDLE_W) / (GPU_MAX_W - GPU_IDLE_W));
        v->act[i] += (pw - v->act[i]) * kn;
        v->power[i] += (s->power[i] - v->power[i]) * kn;
    }
    v->rx += (logn(y->net_rx + y->ts_rx, 2e4, 6e8) - v->rx) * kn;
    v->tx += (logn(y->net_tx + y->ts_tx, 2e4, 3e8) - v->tx) * kn;
    v->rd += (logn(y->disk_read, 2e6, 7e9) - v->rd) * kn;
    v->wr += (logn(y->disk_write, 2e6, 5e9) - v->wr) * kn;
    double tok = gpu_source ? 0 : s->tok_s;
    v->tok += (tok - v->tok) * kn;
    double kc = fmin(1, dt * 1.2);
    for (int i = 0; i < MAX_CPUS; i++)
        v->clk[i] += (clamp01((y->freq[i] - 600) / 5100) - v->clk[i]) * kc;
    v->ctxt += (logn(y->ctxt_s, 1e4, 3e6) - v->ctxt) * kn;
    v->forks += (logn(y->forks_s, 4, 4000) - v->forks) * kn;
    v->tcp += (logn(y->tcp_inuse, 10, 2000) - v->tcp) * kn;
    double w = isnan(y->cpu_watts) ? 35 + 140 * y->cpu_avg : y->cpu_watts;   /* no RAPL: estimate */
    v->cpuw += (clamp01((w - 30) / 190) - v->cpuw) * kn;
    for (int i = 0; i < N_GPUS; i++)
        v->fan[i] += (clamp01(s->fan[i]) - v->fan[i]) * kn;
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
    view_t v = { 0 };
    double last, next_poll = 0, t0;
    int fd = -1, bench = 0, showcase = 0, active = 0;

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
        /* showcase moments: flat out, a working evening, asleep */
        struct { double at; const char *png; } scenes[] = {
            { 19.5, "skyline_preview.png" }, { 9.5, "skyline_mid.png" }, { 2.0, "skyline_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(5 + k);
            memset(puffs, 0, sizeof(puffs));
            memset(cars, 0, sizeof(cars));
            memset(trains, 0, sizeof(trains));
            memset(&v, 0, sizeof(v));
            hud_key[0] = 0;
            int n = 12 * FPS_BUSY;
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                if (i == n - 60)
                    b0 = now_s();
                double t = scenes[k].at - 10 + i / (double)FPS_BUSY;
                showcase_poll(&y, &s, scenes[k].at);
                ease_view(&v, &y, &s, i < n - 100 ? 0.5 : 1.0 / FPS_BUSY);
                active = simulate(&v, 1.0 / FPS_BUSY, t);
                render(cr, &v, t);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes, %d cars+trains\n", scenes[k].png,
                   (now_s() - b0) * 1000 / 60, len, active);
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
            showcase_poll(&y, &s, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 0.5;
            if (demo) {
                demo_poll(&y, &s, t - t0);
            } else {
                sys_poll(&y, t);
                gpus_poll(&s);
                if (gpu_source)
                    gpu_rate_poll(&s);
                else
                    vllm_poll(&s, t);
            }
        }
        /* start from the real picture, not a fade in (the first real poll has no rates yet) */
        ease_view(&v, &y, &s, first ? 10 : dt);
        if (first && (y.cpu_temp > 0 || demo))
            first = 0;
        active = simulate(&v, dt, t - t0);
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

        /* Slow down when the city is asleep */
        int idle = v.cpu_avg < 0.08 && v.act[0] + v.act[1] < 0.15 && v.tok < 1 && active < 3 && !showcase;
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
