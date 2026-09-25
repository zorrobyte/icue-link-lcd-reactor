/*
 * weather: machine weather on the iCUE LINK AIO pump LCD.
 *
 * A small planet seen from orbit, turning slowly in the dark, whose weather is the
 * state of the machine. CPU temperature is the climate: ice caps shrink, oceans turn
 * from deep blue to amber and deserts spread as it heats up. Total CPU load is cloud
 * cover and the busiest CPU threads each spin up a storm. RAM use is how dense and
 * humid the clouds are. Each NVMe drive owns a continent and rains and throws
 * lightning over it with its I/O. Network traffic is wind: jet-stream streaks, cyan
 * blowing west for download, gold blowing east for upload. The two GPUs are the two
 * auroras, GPU 0 at the north pole and GPU 1 at the south, bright with their power.
 * Interrupts are lightning inside the storms, the CPU clock is how fast the clouds
 * drift, major page faults are meteors burning up in the atmosphere, CPU package
 * power is the sun's glare, and pressure stall (PSI) is the barometer: "986 hPa LOW".
 * The day count is the machine's uptime.
 *
 * Everything is procedural: the planet, its clouds and the storms are generated
 * once at start-up from noise, and the globe is drawn per pixel from those maps.
 * Run with --demo to simulate data, --showcase for a scripted 36 s arc from calm to
 * a maxed-out machine and back, --bench to write preview PNGs.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <cairo/cairo.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/hidraw.h>
#include <math.h>
#include <netinet/in.h>
#include <nvml.h>
#include <signal.h>
#include <stdint.h>
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

/* ---- system sensors ---- */

/*
 * CPU per-thread load and temperature, motherboard temps, RAM, DIMM and NVMe temps,
 * NVMe throughput and network throughput, all from /proc and /sys. hwmon devices are
 * found by name, never by number. sys_poll() differences the counters since the last
 * call, so call it at a steady ~1-2 Hz. Missing sensors read as NAN (temps) or 0.
 */
#define SYS_MAX_CPUS    64
#define SYS_N_NVME      3
#define SYS_N_DIMM      2

typedef struct {
    int    n_cpus;
    double cpu[SYS_MAX_CPUS];               /* per hardware thread, 0..1 */
    double cpu_total;                       /* 0..1 */
    double cpu_temp;                        /* k10temp Tctl, C */
    double board_cpu, board_pkg, board_mb, board_tsensor, board_vrm;  /* asusec, C */
    double fan_rpm;                         /* asusec fan1 */
    double ram_total_gb;
    double ram_used;                        /* 0..1: 1 - MemAvailable / MemTotal */
    double ram_cached;                      /* 0..1 of MemTotal */
    double dimm_temp[SYS_N_DIMM];           /* spd5118, C */
    double nvme_temp[SYS_N_NVME];           /* C, [i] = nvme<i> */
    double nvme_read[SYS_N_NVME];           /* bytes/s, [i] = nvme<i>n1 */
    double nvme_write[SYS_N_NVME];
    double net_rx, net_tx;                  /* main ethernet, bytes/s */
    double vpn_rx, vpn_tx;                  /* tailscale0, bytes/s */
    double cpu_mhz[SYS_MAX_CPUS];           /* per-thread clock (scaling_cur_freq) */
    double cpu_mhz_avg;
    double ctxt_s, intr_s, forks_s;         /* context switches, interrupts, new processes per second */
    int    procs_running, tasks;            /* runnable now (/proc/stat), total tasks (/proc/loadavg) */
    double loadavg1;
    double psi_cpu, psi_io, psi_mem, psi_irq;   /* PSI "some avg10" (irq: "full"), stall % */
    double pgfault_s, pgmajfault_s;         /* page faults per second, and major (disk/swap) ones */
    double swapin_s, swapout_s;             /* pages per second */
    int    tcp_inuse;                       /* open TCP sockets */
    int    entropy;                         /* entropy_avail, bits */
    double uptime_s;
    double cpu_watts;                       /* package power from RAPL, NAN if not readable (root only) */
} sys_stats;

static const char *sys_disk[SYS_N_NVME] = { "nvme0n1", "nvme1n1", "nvme2n1" };
static const char *sys_net_main = "enp12s0", *sys_net_vpn = "tailscale0";
static char sys_k10[64], sys_asus[64], sys_dimm[SYS_N_DIMM][64], sys_nvme_hw[SYS_N_NVME][64];

static int sys_read(const char *path, char *buf, int cap)
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

/* hwmon millidegrees (or plain integer for fans) from dir/file, NAN if missing */
static double sys_hwmon(const char *dir, const char *file, double scale)
{
    char path[128], buf[32];
    if (!dir[0])
        return NAN;
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    if (sys_read(path, buf, sizeof(buf)) <= 0)
        return NAN;
    return atof(buf) * scale;
}

static void sys_init(void)
{
    DIR *d = opendir("/sys/class/hwmon");
    struct dirent *e;
    int ndimm = 0;

    if (!d)
        return;
    while ((e = readdir(d))) {
        char dir[64], path[96], name[32], link[256];
        if (strncmp(e->d_name, "hwmon", 5) != 0)
            continue;
        snprintf(dir, sizeof(dir), "/sys/class/hwmon/%.40s", e->d_name);
        snprintf(path, sizeof(path), "%s/name", dir);
        if (sys_read(path, name, sizeof(name)) <= 0)
            continue;
        name[strcspn(name, "\n")] = 0;
        if (!strcmp(name, "k10temp"))
            strcpy(sys_k10, dir);
        else if (!strcmp(name, "asusec"))
            strcpy(sys_asus, dir);
        else if (!strcmp(name, "spd5118") && ndimm < SYS_N_DIMM)
            strcpy(sys_dimm[ndimm++], dir);
        else if (!strcmp(name, "nvme")) {
            /* hwmonN/device -> .../nvme/nvmeK: controller K */
            snprintf(path, sizeof(path), "%s/device", dir);
            ssize_t n = readlink(path, link, sizeof(link) - 1);
            if (n > 0) {
                link[n] = 0;
                const char *base = strrchr(link, '/');
                base = base ? base + 1 : link;
                if (!strncmp(base, "nvme", 4) && isdigit((unsigned char)base[4])) {
                    int k = atoi(base + 4);
                    if (k < SYS_N_NVME)
                        strcpy(sys_nvme_hw[k], dir);
                }
            }
        }
    }
    closedir(d);
    if (ndimm == 2 && strcmp(sys_dimm[0], sys_dimm[1]) > 0) {     /* stable order */
        char tmp[64];
        strcpy(tmp, sys_dimm[0]);
        strcpy(sys_dimm[0], sys_dimm[1]);
        strcpy(sys_dimm[1], tmp);
    }
}

static double sys_rate(unsigned long long now, unsigned long long before, double mult, double dt)
{
    return now >= before && dt > 0 ? (now - before) * mult / dt : 0;
}

static void sys_poll(sys_stats *s, double t)
{
    static unsigned long long cpu_idle[SYS_MAX_CPUS + 1], cpu_all[SYS_MAX_CPUS + 1];
    static unsigned long long disk_rd[SYS_N_NVME], disk_wr[SYS_N_NVME], net[4], kstat[3], vmst[4];
    static double last_t;
    static int have;
    static char buf[32768];
    double dt = have ? t - last_t : 0;

    /* CPU: jiffies per thread, differenced between polls; then the kernel counters */
    if (sys_read("/proc/stat", buf, sizeof(buf)) > 0) {
        char *p = buf;
        int n = 0;
        while (!strncmp(p, "cpu", 3)) {
            int idx = isdigit((unsigned char)p[3]) ? atoi(p + 3) : -1;
            int slot = idx < 0 ? SYS_MAX_CPUS : idx;
            unsigned long long v[8] = { 0 };
            char *sp = strchr(p, ' ');
            if (sp && slot <= SYS_MAX_CPUS &&
                sscanf(sp, "%llu %llu %llu %llu %llu %llu %llu %llu",
                       &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) >= 4) {
                unsigned long long idle = v[3] + v[4], all = 0;
                for (int i = 0; i < 8; i++)
                    all += v[i];
                double load = 0;
                if (have && all > cpu_all[slot])
                    load = clamp01(1.0 - (double)(idle - cpu_idle[slot]) / (double)(all - cpu_all[slot]));
                if (idx < 0)
                    s->cpu_total = load;
                else if (idx < SYS_MAX_CPUS) {
                    s->cpu[idx] = load;
                    if (idx + 1 > n)
                        n = idx + 1;
                }
                cpu_idle[slot] = idle;
                cpu_all[slot] = all;
            }
            p = strchr(p, '\n');
            if (!p)
                break;
            p++;
        }
        s->n_cpus = n;
        const char *keys[3] = { "\nctxt ", "\nintr ", "\nprocesses " };
        double *out[3] = { &s->ctxt_s, &s->intr_s, &s->forks_s };
        for (int i = 0; i < 3; i++) {
            char *q = strstr(buf, keys[i]);
            if (!q)
                continue;
            unsigned long long v = strtoull(q + strlen(keys[i]), NULL, 10);
            *out[i] = have ? sys_rate(v, kstat[i], 1, dt) : 0;
            kstat[i] = v;
        }
        char *q = strstr(buf, "\nprocs_running ");
        if (q)
            s->procs_running = atoi(q + 15);
    }

    /* Clocks */
    double mhz = 0;
    for (int i = 0; i < s->n_cpus; i++) {
        char path[80], v[32];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        s->cpu_mhz[i] = sys_read(path, v, sizeof(v)) > 0 ? atof(v) / 1000 : 0;
        mhz += s->cpu_mhz[i];
    }
    s->cpu_mhz_avg = s->n_cpus ? mhz / s->n_cpus : 0;

    /* Load average and task counts: "0.52 0.61 0.70 2/3800 12345" */
    if (sys_read("/proc/loadavg", buf, sizeof(buf)) > 0) {
        int tasks = 0;
        if (sscanf(buf, "%lf %*f %*f %*d/%d", &s->loadavg1, &tasks) == 2)
            s->tasks = tasks;
    }

    /* Pressure stall information */
    {
        const char *names[4] = { "cpu", "io", "memory", "irq" };
        double *out[4] = { &s->psi_cpu, &s->psi_io, &s->psi_mem, &s->psi_irq };
        for (int i = 0; i < 4; i++) {
            char path[40];
            snprintf(path, sizeof(path), "/proc/pressure/%s", names[i]);
            *out[i] = 0;
            if (sys_read(path, buf, sizeof(buf)) > 0) {
                char *q = strstr(buf, i == 3 ? "full avg10=" : "some avg10=");
                if (q)
                    *out[i] = atof(q + 11);
            }
        }
    }

    /* Page faults and swap */
    if (sys_read("/proc/vmstat", buf, sizeof(buf)) > 0) {
        const char *keys[4] = { "\npgfault ", "\npgmajfault ", "\npswpin ", "\npswpout " };
        double *out[4] = { &s->pgfault_s, &s->pgmajfault_s, &s->swapin_s, &s->swapout_s };
        for (int i = 0; i < 4; i++) {
            char *q = strstr(buf, keys[i]);
            if (!q)
                continue;
            unsigned long long v = strtoull(q + strlen(keys[i]), NULL, 10);
            *out[i] = have ? sys_rate(v, vmst[i], 1, dt) : 0;
            vmst[i] = v;
        }
    }

    /* Open TCP sockets, entropy, uptime */
    if (sys_read("/proc/net/sockstat", buf, sizeof(buf)) > 0) {
        char *q = strstr(buf, "TCP: inuse ");
        if (q)
            s->tcp_inuse = atoi(q + 11);
    }
    if (sys_read("/proc/sys/kernel/random/entropy_avail", buf, sizeof(buf)) > 0)
        s->entropy = atoi(buf);
    if (sys_read("/proc/uptime", buf, sizeof(buf)) > 0)
        s->uptime_s = atof(buf);

    /* CPU package power: RAPL energy counter (root only; NAN otherwise, and we stop trying) */
    {
        static int rapl_denied;
        static unsigned long long last_uj;
        s->cpu_watts = NAN;
        if (!rapl_denied) {
            int n = sys_read("/sys/class/powercap/intel-rapl:0/energy_uj", buf, sizeof(buf));
            if (n <= 0)
                rapl_denied = 1;
            else {
                unsigned long long uj = strtoull(buf, NULL, 10);
                if (have && uj >= last_uj && dt > 0)
                    s->cpu_watts = (uj - last_uj) / 1e6 / dt;
                last_uj = uj;
            }
        }
    }

    /* RAM */
    if (sys_read("/proc/meminfo", buf, sizeof(buf)) > 0) {
        unsigned long long total = 0, avail = 0, cached = 0;
        char *p;
        if ((p = strstr(buf, "MemTotal:")))
            total = strtoull(p + 9, NULL, 10);
        if ((p = strstr(buf, "MemAvailable:")))
            avail = strtoull(p + 13, NULL, 10);
        if ((p = strstr(buf, "\nCached:")))
            cached = strtoull(p + 8, NULL, 10);
        if (total) {
            s->ram_total_gb = total / 1048576.0;
            s->ram_used = clamp01(1.0 - (double)avail / total);
            s->ram_cached = clamp01((double)cached / total);
        }
    }

    /* NVMe throughput: sectors read (field 6) and written (field 10), 512 bytes each */
    if (sys_read("/proc/diskstats", buf, sizeof(buf)) > 0) {
        char *p = buf;
        while (p && *p) {
            char name[32];
            unsigned long long rd, wr;
            if (sscanf(p, "%*u %*u %31s %*u %*u %llu %*u %*u %*u %llu", name, &rd, &wr) == 3)
                for (int i = 0; i < SYS_N_NVME; i++)
                    if (!strcmp(name, sys_disk[i])) {
                        s->nvme_read[i]  = have ? sys_rate(rd, disk_rd[i], 512, dt) : 0;
                        s->nvme_write[i] = have ? sys_rate(wr, disk_wr[i], 512, dt) : 0;
                        disk_rd[i] = rd;
                        disk_wr[i] = wr;
                    }
            p = strchr(p, '\n');
            if (p)
                p++;
        }
    }

    /* Network */
    if (sys_read("/proc/net/dev", buf, sizeof(buf)) > 0) {
        const char *ifs[2] = { sys_net_main, sys_net_vpn };
        for (int i = 0; i < 2; i++) {
            char key[40];
            unsigned long long rx, tx;
            snprintf(key, sizeof(key), "%s:", ifs[i]);
            char *p = buf;
            while ((p = strstr(p, key)) && p != buf && p[-1] != ' ' && p[-1] != '\n')
                p++;                                    /* whole interface names only */
            if (!p || sscanf(p + strlen(key), "%llu %*u %*u %*u %*u %*u %*u %*u %llu", &rx, &tx) != 2)
                continue;
            double r = have ? sys_rate(rx, net[i * 2], 1, dt) : 0;
            double w = have ? sys_rate(tx, net[i * 2 + 1], 1, dt) : 0;
            if (i == 0)
                s->net_rx = r, s->net_tx = w;
            else
                s->vpn_rx = r, s->vpn_tx = w;
            net[i * 2] = rx;
            net[i * 2 + 1] = tx;
        }
    }

    /* Temperatures */
    s->cpu_temp      = sys_hwmon(sys_k10, "temp1_input", 0.001);
    s->board_cpu     = sys_hwmon(sys_asus, "temp1_input", 0.001);
    s->board_pkg     = sys_hwmon(sys_asus, "temp2_input", 0.001);
    s->board_mb      = sys_hwmon(sys_asus, "temp3_input", 0.001);
    s->board_tsensor = sys_hwmon(sys_asus, "temp4_input", 0.001);
    s->board_vrm     = sys_hwmon(sys_asus, "temp5_input", 0.001);
    s->fan_rpm       = sys_hwmon(sys_asus, "fan1_input", 1);
    for (int i = 0; i < SYS_N_DIMM; i++)
        s->dimm_temp[i] = sys_hwmon(sys_dimm[i], "temp1_input", 0.001);
    for (int i = 0; i < SYS_N_NVME; i++)
        s->nvme_temp[i] = sys_hwmon(sys_nvme_hw[i], "temp1_input", 0.001);

    last_t = t;
    have = 1;
}

/* ---- end system sensors ---- */

/* ---------------------------------------------------------------- simulated data */

static double frand(void) { return rand() / (double)RAND_MAX; }
static double sstep(double a, double b, double x) { x = clamp01((x - a) / (b - a)); return x * x * (3 - 2 * x); }

/*
 * Spread a total CPU load over the threads the way real work does: a few threads
 * much busier than the rest at moderate load, all of them pinned at full load.
 */
static void spread_threads(sys_stats *s, double total, double t)
{
    s->n_cpus = 32;
    double spread = 2.4 * sqrt(clamp01(total) * (1 - clamp01(total))) + 0.12;
    double sum = 0;
    for (int i = 0; i < s->n_cpus; i++) {
        double rank = ((i * 13) % 32) / 31.0;       /* fixed ordering of busy threads */
        double wob = 0.06 * sin(t * 0.9 + i * 2.1) + 0.04 * sin(t * 2.3 + i);
        s->cpu[i] = clamp01(total + (0.5 - rank) * spread + wob * (0.3 + total));
        sum += s->cpu[i];
    }
    s->cpu_total = sum / s->n_cpus;
}

/* --demo: everything wanders on its own slow cycle */
static void demo_poll(sys_stats *s, stats *g, double t)
{
    double busy = clamp01(0.45 + 0.5 * sin(t * 0.19) + 0.2 * sin(t * 0.53 + 1));
    spread_threads(s, busy, t);
    s->cpu_temp = 44 + 48 * clamp01(0.5 + 0.55 * sin(t * 0.19 - 0.9));
    s->ram_total_gb = 91;
    s->ram_used = 0.3 + 0.6 * (0.5 + 0.5 * sin(t * 0.07 + 2));
    for (int d = 0; d < SYS_N_NVME; d++) {
        double v = sin(t * 0.31 + d * 2.1) * sin(t * 0.11 + d);
        s->nvme_read[d] = v > 0.25 ? pow(10, 6 + 3.5 * (v - 0.25) / 0.75) : 2e4;
        s->nvme_write[d] = s->nvme_read[d] * 0.3;
    }
    s->net_rx = pow(10, 3.5 + 4.5 * (0.5 + 0.5 * sin(t * 0.23 + 0.5)));
    s->net_tx = pow(10, 3.0 + 4.3 * (0.5 + 0.5 * sin(t * 0.29 + 2.0)));
    s->cpu_mhz_avg = 2600 + 2800 * busy;
    s->intr_s = pow(10, 4.8 + 1.5 * busy);
    s->pgmajfault_s = pow(10, 1 + 4.2 * (0.5 + 0.5 * sin(t * 0.17 + 3)));
    s->psi_cpu = 45 * pow(busy, 3);
    s->psi_mem = 12 * pow(clamp01(s->ram_used - 0.5) * 2, 2);
    s->cpu_watts = 35 + 170 * busy;
    s->uptime_s = 3.2 * 86400 + t;
    for (int i = 0; i < N_GPUS; i++) {
        g->power[i] = 30 + 530 * clamp01(0.4 + 0.7 * sin(t * 0.13 + i * 2.4));
        g->load[i] = clamp01((g->power[i] - 30) / 400);
        g->temp[i] = (int)(35 + 35 * g->load[i]);
    }
}

/*
 * --showcase: a scripted 36 s arc for filming or GIFs. A calm, cold planet; the CPU
 * wakes and clouds build; the drives start and storms break over the continents; the
 * network picks up; everything maxes out (heatwave, hurricanes, both auroras blazing);
 * then it all winds back down to calm.
 */
#define SC_LEN 36.0
static const double sc_t[]    = { 0,    5,    9,    13,   17,   21,   27,   31,   34,   36 };
static const double sc_load[] = { .02,  .03,  .30,  .45,  .70,  1.0,  1.0,  .25,  .03,  .02 };
static const double sc_temp[] = { 40,   40,   52,   60,   71,   92,   95,   68,   44,   40 };
static const double sc_ram[]  = { .22,  .22,  .38,  .50,  .66,  .92,  .92,  .55,  .25,  .22 };
static const double sc_d0[]   = { 4,    4,    4,    8.6,  9.0,  9.4,  9.5,  6.5,  4,    4 };   /* log10 bytes/s */
static const double sc_d1[]   = { 4,    4,    4,    4,    8.8,  9.3,  9.6,  4,    4,    4 };
static const double sc_d2[]   = { 4,    4,    8.2,  5,    8.4,  9.2,  9.5,  4,    4,    4 };
static const double sc_rx[]   = { 3.3,  3.4,  6.0,  7.4,  7.9,  8.5,  8.6,  6.8,  3.6,  3.3 };
static const double sc_tx[]   = { 3.0,  3.2,  5.0,  6.2,  7.6,  8.3,  8.4,  5.5,  3.3,  3.0 };
static const double sc_g0[]   = { 30,   30,   45,   140,  300,  560,  570,  150,  35,   30 };
static const double sc_g1[]   = { 32,   32,   35,   90,   250,  540,  560,  90,   35,   32 };
static const double sc_mhz[]  = { 3000, 3100, 4600, 5000, 5300, 5450, 5450, 4200, 3100, 3000 };
static const double sc_intr[] = { 4.9,  4.9,  5.4,  5.6,  5.9,  6.3,  6.4,  5.5,  4.95, 4.9 };   /* log10 /s */
static const double sc_maj[]  = { 1.0,  1.0,  1.5,  2.6,  3.6,  4.6,  5.3,  3.0,  1.0,  1.0 };   /* log10 /s */
static const double sc_psi[]  = { 0.2,  0.2,  2,    6,    14,   38,   45,   8,    0.5,  0.2 };
static const double sc_watt[] = { 38,   38,   90,   120,  160,  205,  210,  90,   40,   38 };
#define SC_N (int)(sizeof(sc_t) / sizeof(sc_t[0]))

static double sc_key(const double *k, double u)
{
    for (int i = 0; i < SC_N - 1; i++)
        if (u < sc_t[i + 1])
            return k[i] + (k[i + 1] - k[i]) * sstep(sc_t[i], sc_t[i + 1], u);
    return k[SC_N - 1];
}

static void showcase_poll(sys_stats *s, stats *g, double t)
{
    double u = fmod(t, SC_LEN);
    spread_threads(s, sc_key(sc_load, u), t);
    s->cpu_temp = sc_key(sc_temp, u);
    s->ram_total_gb = 91;
    s->ram_used = sc_key(sc_ram, u);
    const double *dk[SYS_N_NVME] = { sc_d0, sc_d1, sc_d2 };
    for (int d = 0; d < SYS_N_NVME; d++) {
        s->nvme_read[d] = pow(10, sc_key(dk[d], u) + 0.15 * sin(t * 3 + d));
        s->nvme_write[d] = 0;
    }
    s->net_rx = pow(10, sc_key(sc_rx, u));
    s->net_tx = pow(10, sc_key(sc_tx, u));
    s->cpu_mhz_avg = sc_key(sc_mhz, u);
    s->intr_s = pow(10, sc_key(sc_intr, u));
    s->pgmajfault_s = pow(10, sc_key(sc_maj, u));
    s->psi_cpu = sc_key(sc_psi, u);
    s->cpu_watts = sc_key(sc_watt, u);
    s->uptime_s = 12.4 * 86400 + t;
    g->power[0] = sc_key(sc_g0, u);
    g->power[1] = sc_key(sc_g1, u);
    for (int i = 0; i < N_GPUS; i++) {
        g->load[i] = clamp01((g->power[i] - 30) / 400);
        g->temp[i] = (int)(35 + 35 * g->load[i]);
    }
}

/* ---------------------------------------------------------------- planet */

#define FPS_BUSY        20
#define FPS_IDLE        15
#define CX              240
#define CY              210
#define PR              158         /* planet radius, px */
#define TW              1024        /* equirectangular maps, TW x TH */
#define TH              512
#define TWM             (TW - 1)
#define SEA             128         /* height byte at sea level */
#define TEMP_COOL       42.0        /* Tctl for the coldest climate */
#define TEMP_HOT        95.0        /* Tctl for the hottest */
#define CLIMATE_STEPS   64
#define DAY_S           90.0        /* one turn of the planet */
#define MAX_STORMS      8
#define STORM_MIN_LOAD  0.30        /* a thread needs this much load to brew a storm */
#define N_JETS          110
#define MAX_BOLTS       24
#define AUR_N           72
#define MAX_METEORS     16
#define HUD_Y           386
#define HUD_H           84

static const double TILT = 0.07;           /* north pole leans toward us, radians */
static const double ROLL = -0.85;          /* and the axis is turned on screen */
static double M[3][3];                     /* world -> view */
static double SUN[3];                      /* light direction, view space */

/* Three continents, one per NVMe drive: centre lat, lon and radius in degrees */
static const double cont_lat[SYS_N_NVME] = { 22, -18, 30 };
static const double cont_lon[SYS_N_NVME] = { -28, 88, -152 };
static const double cont_rad[SYS_N_NVME] = { 36, 32, 30 };

static uint32_t *tex;                      /* surface colours for the current climate; alpha byte = ocean */
static uint8_t *h_map, *m_map, *ice_map, *rel_map, *rain_map, *cloud_a, *cloud_b;
static int cloud_cdf[256];                 /* cumulative texel count of the combined cloud noise */

typedef struct { uint32_t off, u, row; uint8_t shade, cshade, spec, haze, hlight, cover; } gpx;
static uint8_t *cloud_n;                   /* cloud opacity under each globe pixel, this frame */
static int fb_stride4;                     /* frame stride in pixels */
static gpx *gp;
static int n_gp;

typedef struct { double lat, lon, amp, spin; } storm;
typedef struct { double lat, lon, spd, age, life, len, wob; int dir, alive; } jet;
typedef struct { double lat, lon, age, life, ang; unsigned seed; int alive, cloud; } bolt;

/* Eased drivers: everything the picture shows, smoothed so nothing jumps */
typedef struct {
    double temp, climate, load, ram, cpu[SYS_MAX_CPUS], disk[SYS_N_NVME], rx, tx, aur[N_GPUS];
    double clock, intr, meteors, sun, psi;
} wx_t;
typedef struct { double x, y, vx, vy, age, life; int alive; } meteor;

static wx_t wx;
static storm storms[SYS_MAX_CPUS];
static jet jets[N_JETS];
static bolt bolts[MAX_BOLTS];
static meteor meteors[MAX_METEORS];
static double meteor_acc, storm_acc, uptime_days, pressure_hpa = 1013;
static double theta_s, theta_a, theta_b, strike_acc[SYS_N_NVME];
static int climate_step = -1, n_threads = 32;
static uint8_t cloud_lut[256];
static int atm_rgb[3], storm_rgb[3];
static cairo_surface_t *stars_img, *bg_img, *storm_spr, *hud_cache, *sun_img;
static char hud_key[128];

/* --- noise --- */

static int perm[512];

static void noise_init(unsigned seed)
{
    srand(seed);
    for (int i = 0; i < 256; i++)
        perm[i] = i;
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1), tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    for (int i = 0; i < 256; i++)
        perm[256 + i] = perm[i];
}

static double nfade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }
static double nlerp(double t, double a, double b) { return a + t * (b - a); }
static double ngrad(int h, double x, double y, double z)
{
    h &= 15;
    double u = h < 8 ? x : y, v = h < 4 ? y : (h == 12 || h == 14) ? x : z;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

static double noise3(double x, double y, double z)
{
    double fx = floor(x), fy = floor(y), fz = floor(z);
    int X = (int)fx & 255, Y = (int)fy & 255, Z = (int)fz & 255;
    x -= fx, y -= fy, z -= fz;
    double u = nfade(x), v = nfade(y), w = nfade(z);
    int A = perm[X] + Y, AA = perm[A] + Z, AB = perm[A + 1] + Z;
    int B = perm[X + 1] + Y, BA = perm[B] + Z, BB = perm[B + 1] + Z;
    return nlerp(w, nlerp(v, nlerp(u, ngrad(perm[AA], x, y, z), ngrad(perm[BA], x - 1, y, z)),
                             nlerp(u, ngrad(perm[AB], x, y - 1, z), ngrad(perm[BB], x - 1, y - 1, z))),
                    nlerp(v, nlerp(u, ngrad(perm[AA + 1], x, y, z - 1), ngrad(perm[BA + 1], x - 1, y, z - 1)),
                             nlerp(u, ngrad(perm[AB + 1], x, y - 1, z - 1), ngrad(perm[BB + 1], x - 1, y - 1, z - 1))));
}

static double fbm(double x, double y, double z, int oct)
{
    double sum = 0, amp = 0.5;
    for (int i = 0; i < oct; i++) {
        sum += amp * noise3(x, y, z);
        x = x * 2.03 + 17.1, y = y * 2.03 + 3.7, z = z * 2.03 + 9.3;
        amp *= 0.5;
    }
    return sum;
}

/* --- geometry --- */

static void view_matrix(void)
{
    double ct = cos(TILT), st = sin(TILT), cr = cos(ROLL), sr = sin(ROLL);
    /* M = Rz(roll) * Rx(tilt); Rx turns +Y (north) toward the viewer (+Z) */
    double rx[3][3] = { { 1, 0, 0 }, { 0, ct, -st }, { 0, st, ct } };
    double rz[3][3] = { { cr, -sr, 0 }, { sr, cr, 0 }, { 0, 0, 1 } };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M[i][j] = 0;
            for (int k = 0; k < 3; k++)
                M[i][j] += rz[i][k] * rx[k][j];
        }
    double sx = -0.52, sy = 0.44, sz = 0.73, n = sqrt(sx * sx + sy * sy + sz * sz);
    SUN[0] = sx / n, SUN[1] = sy / n, SUN[2] = sz / n;
}

/* Unit vector for a world lat/lon (radians), in view space (x right, y up, z to viewer) */
static void to_view(double lat, double lon, double v[3])
{
    double p[3] = { cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon) };
    for (int i = 0; i < 3; i++)
        v[i] = M[i][0] * p[0] + M[i][1] * p[1] + M[i][2] * p[2];
}

static double light_at(const double n[3], double lo, double hi)
{
    return sstep(lo, hi, n[0] * SUN[0] + n[1] * SUN[1] + n[2] * SUN[2]);
}

/* Angle between two unit vectors given by lat/lon (radians) */
static double ang_dist(double la1, double lo1, double la2, double lo2)
{
    double c = sin(la1) * sin(la2) + cos(la1) * cos(la2) * cos(lo1 - lo2);
    return acos(fmax(-1, fmin(1, c)));
}

/* Large cloud shapes plus fine detail, which is strongest right at their edges */
static inline int cloud_mix(int a, int b)
{
    int n = a + ((b - 128) * 7 >> 4);
    return n < 0 ? 0 : n > 255 ? 255 : n;
}

/* --- the world, generated once --- */

static void build_world(void)
{
    const double D = M_PI / 180;
    float *hf = malloc(sizeof(float) * TW * TH);
    float *cf = malloc(sizeof(float) * TW * TH), *df = malloc(sizeof(float) * TW * TH);
    h_map = malloc(TW * TH);
    m_map = malloc(TW * TH);
    ice_map = malloc(TW * TH);
    rel_map = malloc(TW * TH);
    rain_map = malloc(TW * TH);
    cloud_a = malloc(TW * TH);
    cloud_b = malloc(TW * TH);
    tex = malloc(sizeof(uint32_t) * TW * TH);
    noise_init(1234);

    double cmin = 1e9, cmax = -1e9, dmin = 1e9, dmax = -1e9;
    for (int v = 0; v < TH; v++) {
        double lat = (0.5 - (v + 0.5) / TH) * M_PI, alat = fabs(lat) / D;
        for (int u = 0; u < TW; u++) {
            int i = v * TW + u;
            double lon = ((u + 0.5) / TW - 0.5) * 2 * M_PI;
            double x = cos(lat) * sin(lon), y = sin(lat), z = cos(lat) * cos(lon);

            /* Land: three warped continental blobs plus island noise */
            double wx_ = fbm(x * 1.6 + 5.2, y * 1.6, z * 1.6, 3), wy_ = fbm(x * 1.6, y * 1.6 + 7.7, z * 1.6, 3);
            double wz_ = fbm(x * 1.6, y * 1.6, z * 1.6 + 2.9, 3);
            double qx = x + 0.42 * wx_, qy = y + 0.42 * wy_, qz = z + 0.42 * wz_;
            double qn = sqrt(qx * qx + qy * qy + qz * qz);
            double qlat = asin(qy / qn), qlon = atan2(qx, qz);
            double h = 0.33 + 0.36 * fbm(qx * 2.1, qy * 2.1, qz * 2.1, 7);
            double land = 0;
            for (int k = 0; k < SYS_N_NVME; k++) {
                double d = ang_dist(qlat, qlon, cont_lat[k] * D, cont_lon[k] * D) / D;
                land = fmax(land, sstep(cont_rad[k] * 1.3, cont_rad[k] * 0.35, d));
            }
            h += 0.26 * land;
            double ridge = 1 - fabs(fbm(qx * 4.5 + 1, qy * 4.5, qz * 4.5, 4) * 1.6);
            h += 0.12 * ridge * ridge * land;
            hf[i] = (float)h;

            /* Moisture: wet tropics and coasts, dry subtropics */
            double m = 0.5 + 0.55 * fbm(x * 2.6 + 9, y * 2.6, z * 2.6, 4);
            m += 0.20 * exp(-pow(alat / 12, 2)) - 0.22 * exp(-pow((alat - 25) / 10, 2));
            m += 0.10 * exp(-pow((alat - 55) / 12, 2));
            m_map[i] = (uint8_t)(255 * clamp01(m));

            /* How polar a texel is, in degrees, with a ragged edge */
            double ice = alat + 7 * fbm(x * 5, y * 5 + 3, z * 5, 3) * 2;
            ice_map[i] = (uint8_t)fmin(255, fmax(0, ice * 2.5));

            /* Rain regions over each continent (surface space) */
            int best = 0;
            double bw = 0;
            for (int k = 0; k < SYS_N_NVME; k++) {
                double d = ang_dist(lat, lon, cont_lat[k] * D, cont_lon[k] * D) / D;
                double w = sstep(cont_rad[k] * 1.05, cont_rad[k] * 0.3, d);
                if (w > bw)
                    bw = w, best = k + 1;
            }
            rain_map[i] = bw > 0.01 ? (uint8_t)((best << 6) | (int)(bw * 63)) : 0;

            /* Clouds: stretched along the latitudes, banded like a real planet */
            double ax = x + 0.45 * fbm(x * 2 + 3, y * 2, z * 2, 3), ay = y + 0.25 * fbm(x * 2, y * 2 + 1, z * 2, 3);
            double az = z + 0.45 * fbm(x * 2, y * 2, z * 2 + 7, 3);
            double c = fbm(ax * 1.9, ay * 5.0, az * 1.9, 6);
            c += 0.10 * exp(-pow(alat / 8, 2)) + 0.12 * exp(-pow((alat - 52) / 12, 2)) - 0.08 * exp(-pow((alat - 25) / 9, 2));
            cf[i] = (float)c;
            double dd = fbm(x * 6 + 2, y * 13, z * 6, 5);
            df[i] = (float)dd;
            cmin = fmin(cmin, c), cmax = fmax(cmax, c), dmin = fmin(dmin, dd), dmax = fmax(dmax, dd);
        }
    }
    /* Heights around a fixed sea level (0.5 -> SEA) */
    for (int i = 0; i < TW * TH; i++)
        h_map[i] = (uint8_t)(255 * clamp01((hf[i] - 0.5) * 1.5 + 0.5));
    /* Relief shading from the height gradient, light from the north-west */
    for (int v = 0; v < TH; v++)
        for (int u = 0; u < TW; u++) {
            int vn = v > 0 ? v - 1 : v, vs = v < TH - 1 ? v + 1 : v;
            double gx = hf[v * TW + ((u + 1) & TWM)] - hf[v * TW + ((u - 1) & TWM)];
            double gy = hf[vs * TW + u] - hf[vn * TW + u];
            rel_map[v * TW + u] = (uint8_t)fmax(0, fmin(255, 128 - (gx + gy) * 900));
        }
    memset(cloud_cdf, 0, sizeof(cloud_cdf));
    for (int i = 0; i < TW * TH; i++) {
        cloud_a[i] = (uint8_t)(255 * (cf[i] - cmin) / (cmax - cmin));
        cloud_b[i] = (uint8_t)(255 * (df[i] - dmin) / (dmax - dmin));
        cloud_cdf[cloud_mix(cloud_a[i], cloud_b[i])]++;
    }
    for (int i = 1; i < 256; i++)
        cloud_cdf[i] += cloud_cdf[i - 1];
    free(hf);
    free(cf);
    free(df);
}

typedef struct { double r, g, b; } col3;

static col3 mix3(col3 a, col3 b, double t) { return (col3){ a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t }; }

/* A colour at three points of the climate: cold, temperate and hot */
static col3 climate_col(double c, col3 cold, col3 mid, col3 hot)
{
    return c < 0.5 ? mix3(cold, mid, sstep(0, 0.5, c)) : mix3(mid, hot, sstep(0.5, 1, c));
}

/* Repaint the surface for a climate: 0 cold .. 1 hot */
static void recolor(double c)
{
    static uint32_t pal[256][32];
    /* oceans: deep blue, warm tropical turquoise, a jade bloom, then amber */
    col3 deep    = c < 0.72 ? climate_col(c / 0.72, (col3){ 6, 26, 72 }, (col3){ 6, 52, 92 }, (col3){ 0, 78, 92 })
                 : c < 0.84 ? mix3((col3){ 0, 78, 92 }, (col3){ 50, 92, 56 }, sstep(0.72, 0.84, c))
                            : mix3((col3){ 50, 92, 56 }, (col3){ 104, 50, 12 }, sstep(0.84, 1, c));
    col3 shallow = c < 0.72 ? climate_col(c / 0.72, (col3){ 22, 96, 158 }, (col3){ 28, 140, 160 }, (col3){ 40, 184, 168 })
                 : c < 0.84 ? mix3((col3){ 40, 184, 168 }, (col3){ 150, 196, 104 }, sstep(0.72, 0.84, c))
                            : mix3((col3){ 150, 196, 104 }, (col3){ 222, 140, 44 }, sstep(0.84, 1, c));
    col3 wet     = climate_col(c, (col3){ 34, 84, 46 },   (col3){ 64, 96, 40 },   (col3){ 110, 66, 34 });
    col3 dry     = climate_col(c, (col3){ 112, 116, 76 }, (col3){ 156, 136, 82 }, (col3){ 168, 80, 42 });
    col3 sand    = climate_col(c, (col3){ 186, 170, 130 },(col3){ 212, 172, 108 },(col3){ 206, 96, 50 });
    col3 rock    = climate_col(c, (col3){ 104, 100, 96 }, (col3){ 118, 102, 86 }, (col3){ 84, 52, 40 });
    col3 snow    = { 236, 240, 246 };
    double desert = 0.20 + 0.52 * c, snow_e = 0.74 + 0.4 * c;

    for (int h = 0; h < 256; h++)
        for (int mi = 0; mi < 32; mi++) {
            col3 col;
            int ocean = h < SEA;
            if (ocean) {
                double depth = (SEA - h) / (double)SEA;
                col = mix3(shallow, deep, sstep(0.0, 0.45, depth));
            } else {
                double e = (h - SEA) / (255.0 - SEA), m = (mi + 0.5) / 32;
                col = m < desert ? mix3(sand, dry, sstep(desert - 0.14, desert, m))
                                 : mix3(dry, wet, sstep(desert, desert + 0.22, m));
                col = mix3(col, rock, sstep(0.40, 0.75, e) * 0.75);
                col = mix3(col, snow, sstep(snow_e, snow_e + 0.08, e));
                col = mix3(col, sand, sstep(0.03, 0.0, e) * 0.5);   /* beaches */
            }
            pal[h][mi] = (uint32_t)(ocean ? 255 : 0) << 24 | (uint32_t)col.r << 16 | (uint32_t)col.g << 8 | (uint32_t)col.b;
        }

    uint8_t ice_a[256];
    double ice_lat = 61 + 31 * c;
    for (int i = 0; i < 256; i++)
        ice_a[i] = (uint8_t)(255 * sstep(ice_lat - 2.5, ice_lat + 2.5, i / 2.5));

    for (int i = 0; i < TW * TH; i++) {
        uint32_t p = pal[h_map[i]][m_map[i] >> 3];
        int r = p >> 16 & 255, g = p >> 8 & 255, b = p & 255, oc = p >> 24;
        if (!oc) {
            int rel = rel_map[i];
            r = r * rel >> 7, g = g * rel >> 7, b = b * rel >> 7;
        }
        int ia = ice_a[ice_map[i]];
        if (ia) {
            int ir = 226, ig = 236, ib = 248;
            if (!oc) {
                int rel = rel_map[i];
                ir = ir * (rel + 384) >> 9, ig = ig * (rel + 384) >> 9, ib = ib * (rel + 384) >> 9;
            }
            r += (ir - r) * ia >> 8, g += (ig - g) * ia >> 8, b += (ib - b) * ia >> 8;
            oc = oc * (255 - ia) >> 8;
        }
        r = r > 255 ? 255 : r, g = g > 255 ? 255 : g, b = b > 255 ? 255 : b;
        tex[i] = (uint32_t)oc << 24 | (uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b;
    }

    col3 atm = climate_col(c, (col3){ 88, 156, 255 }, (col3){ 110, 176, 240 }, (col3){ 255, 164, 92 });
    atm_rgb[0] = (int)atm.r, atm_rgb[1] = (int)atm.g, atm_rgb[2] = (int)atm.b;
    col3 st = climate_col(c, (col3){ 196, 204, 222 }, (col3){ 200, 204, 216 }, (col3){ 214, 204, 196 });
    storm_rgb[0] = (int)st.r, storm_rgb[1] = (int)st.g, storm_rgb[2] = (int)st.b;
}

/* Per-pixel lookup for the globe: which texel, how lit, how hazy */
static void build_globe(int stride4)
{
    fb_stride4 = stride4;
    cloud_n = calloc((size_t)stride4 * SIZE, 1);
    gp = malloc(sizeof(gpx) * (2 * PR + 6) * (2 * PR + 6));
    double hx = SUN[0], hy = SUN[1], hz = SUN[2] + 1, hn = sqrt(hx * hx + hy * hy + hz * hz);
    hx /= hn, hy /= hn, hz /= hn;
    for (int y = CY - PR - 2; y <= CY + PR + 2; y++)
        for (int x = CX - PR - 2; x <= CX + PR + 2; x++) {
            double dx = (x + 0.5 - CX) / PR, dy = (CY - (y + 0.5)) / PR, d = hypot(dx, dy);
            double cover = clamp01((1 - d) * PR + 0.5);
            if (cover <= 0)
                continue;
            if (d > 0.999)
                dx *= 0.999 / d, dy *= 0.999 / d;
            double n[3] = { dx, dy, sqrt(fmax(0, 1 - dx * dx - dy * dy)) };
            double p[3];
            for (int i = 0; i < 3; i++)                 /* view -> world: M transposed */
                p[i] = M[0][i] * n[0] + M[1][i] * n[1] + M[2][i] * n[2];
            double lat = asin(fmax(-1, fmin(1, p[1]))), lon = atan2(p[0], p[2]);
            double L = n[0] * SUN[0] + n[1] * SUN[1] + n[2] * SUN[2];
            double nh = fmax(0, n[0] * hx + n[1] * hy + n[2] * hz);
            gpx *g = &gp[n_gp++];
            g->off = (uint32_t)(y * stride4 + x);
            g->u = (uint32_t)(fmod(lon / (2 * M_PI) + 1.5, 1.0) * TW * 65536.0);
            int v = (int)((0.5 - lat / M_PI) * TH);
            g->row = (uint32_t)((v < 0 ? 0 : v >= TH ? TH - 1 : v) * TW);
            g->shade  = (uint8_t)(255 * (0.03 + 0.97 * pow(sstep(-0.10, 0.62, L), 0.9)));
            g->cshade = (uint8_t)(255 * (0.04 + 0.96 * sstep(-0.16, 0.50, L)));
            g->spec   = (uint8_t)(255 * clamp01(0.45 * pow(nh, 160) + 0.13 * pow(nh, 14)) * sstep(0.0, 0.3, L));
            g->haze   = (uint8_t)(255 * clamp01(0.05 + 0.92 * pow(1 - n[2], 2.4)));
            g->hlight = (uint8_t)(255 * sstep(-0.30, 0.45, L));
            g->cover  = (uint8_t)(255 * cover);
        }
}

#define SUN_W 250
/* Starfield and a faint galaxy band, drawn once */
static void build_stars(void)
{
    stars_img = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    bg_img = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(stars_img);
    cairo_pattern_t *pat = cairo_pattern_create_radial(CX, CY, PR, CX, CY, 260);
    cairo_pattern_add_color_stop_rgb(pat, 0, 0.022, 0.030, 0.062);
    cairo_pattern_add_color_stop_rgb(pat, 1, 0.004, 0.005, 0.014);
    cairo_set_source(cr, pat);
    cairo_paint(cr);
    cairo_pattern_destroy(pat);

    srand(99);
    /* galaxy band: a diagonal haze of faint dust */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = 0; i < 900; i++) {
        double t = frand() * 1.4 - 0.2, off = (frand() + frand() + frand() - 1.5) * 50;
        double x = 480 * t + off * 0.6, y = 480 * (1 - t) * 0.9 + 40 + off;
        cairo_arc(cr, x, y, 0.4 + frand() * 0.6, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 0.55, 0.60, 0.85, 0.10 + frand() * 0.25);
        cairo_fill(cr);
    }
    for (int i = 0; i < 14; i++) {
        double t = frand(), x = 480 * t, y = 480 * (1 - t) * 0.9 + 40, r = 40 + frand() * 50;
        pat = cairo_pattern_create_radial(x, y, 0, x, y, r);
        cairo_pattern_add_color_stop_rgba(pat, 0, 0.30, 0.26, 0.45, 0.05);
        cairo_pattern_add_color_stop_rgba(pat, 1, 0.30, 0.26, 0.45, 0);
        cairo_set_source(cr, pat);
        cairo_arc(cr, x, y, r, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(pat);
    }
    /* stars */
    for (int i = 0; i < 420; i++) {
        double x = frand() * SIZE, y = frand() * SIZE, m = pow(frand(), 3);
        double tint = frand();
        rgb c = tint < 0.2 ? (rgb){ 1.0, 0.85, 0.7 } : tint < 0.5 ? (rgb){ 0.75, 0.85, 1.0 } : (rgb){ 0.95, 0.95, 1.0 };
        double r = 0.45 + m * 1.1;
        if (m > 0.5) {
            pat = cairo_pattern_create_radial(x, y, 0, x, y, r * 4);
            cairo_pattern_add_color_stop_rgba(pat, 0, c.r, c.g, c.b, 0.25);
            cairo_pattern_add_color_stop_rgba(pat, 1, c.r, c.g, c.b, 0);
            cairo_set_source(cr, pat);
            cairo_arc(cr, x, y, r * 4, 0, 2 * M_PI);
            cairo_fill(cr);
            cairo_pattern_destroy(pat);
        }
        cairo_arc(cr, x, y, r, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, c.r, c.g, c.b, 0.25 + 0.75 * sqrt(m) * frand() + 0.15);
        cairo_fill(cr);
    }
    cairo_destroy(cr);

    /* the sun is off to the upper left: a warm glare, drawn each frame at its brightness */
    sun_img = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SUN_W, SUN_W);
    cr = cairo_create(sun_img);
    pat = cairo_pattern_create_radial(34, 58, 0, 34, 58, SUN_W);
    cairo_pattern_add_color_stop_rgba(pat, 0, 1.0, 0.93, 0.80, 0.34);
    cairo_pattern_add_color_stop_rgba(pat, 0.12, 1.0, 0.88, 0.70, 0.16);
    cairo_pattern_add_color_stop_rgba(pat, 0.45, 1.0, 0.85, 0.65, 0.04);
    cairo_pattern_add_color_stop_rgba(pat, 1, 1.0, 0.85, 0.65, 0);
    cairo_set_source(cr, pat);
    cairo_paint(cr);
    cairo_pattern_destroy(pat);
    cairo_destroy(cr);
}

/* Background for the current climate: stars plus the atmosphere's glow around the limb */
static void build_bg(void)
{
    cairo_surface_flush(stars_img);
    cairo_surface_flush(bg_img);
    int stride = cairo_image_surface_get_stride(bg_img);
    memcpy(cairo_image_surface_get_data(bg_img), cairo_image_surface_get_data(stars_img), (size_t)stride * SIZE);
    uint32_t *px = (uint32_t *)cairo_image_surface_get_data(bg_img);
    const int ext = 34;
    for (int y = CY - PR - ext; y <= CY + PR + ext; y++)
        for (int x = CX - PR - ext; x <= CX + PR + ext; x++) {
            double dx = x + 0.5 - CX, dy = CY - (y + 0.5), r = hypot(dx, dy) - PR;
            if (r < -3 || r > ext || y < 0 || y >= SIZE)
                continue;
            double lit = (dx * SUN[0] + dy * SUN[1]) / (PR + r) / hypot(SUN[0], SUN[1]);
            double k = r < 0 ? 1 : 0.75 * exp(-r / 5.5) + 0.30 * exp(-r / 16);
            k *= 0.30 + 0.70 * sstep(-0.7, 0.8, lit);
            uint32_t *p = &px[y * (stride / 4) + x];
            int c[3] = { *p >> 16 & 255, *p >> 8 & 255, *p & 255 };
            for (int i = 0; i < 3; i++) {
                c[i] += (int)(atm_rgb[i] * k);
                c[i] = c[i] > 255 ? 255 : c[i];
            }
            *p = 0xff000000u | (uint32_t)c[0] << 16 | (uint32_t)c[1] << 8 | (uint32_t)c[2];
        }
    cairo_surface_mark_dirty(bg_img);
}

/* A cyclone: two spiral arms around a dense core with a clear eye, as an alpha mask */
#define STORM_SPR 112
static void build_storm_sprite(void)
{
    storm_spr = cairo_image_surface_create(CAIRO_FORMAT_A8, STORM_SPR, STORM_SPR);
    cairo_surface_flush(storm_spr);
    unsigned char *d = cairo_image_surface_get_data(storm_spr);
    int stride = cairo_image_surface_get_stride(storm_spr);
    for (int y = 0; y < STORM_SPR; y++)
        for (int x = 0; x < STORM_SPR; x++) {
            double dx = (x + 0.5) / (STORM_SPR / 2.0) - 1, dy = (y + 0.5) / (STORM_SPR / 2.0) - 1;
            double r = hypot(dx, dy), a = 0;
            if (r < 1) {
                double th = atan2(dy, dx);
                double ph = th * 2 + 5.0 * log(r + 0.05) + 1.4 * fbm(dx * 2.5, dy * 2.5, 0.5, 3);
                double arms = pow(0.5 + 0.5 * cos(ph), 1.8);
                double fall = pow(1 - r, 1.3);
                double core = exp(-pow(r / 0.26, 2));
                double tex_ = 0.7 + 0.6 * fbm(dx * 6, dy * 6, 2.1, 4);
                a = (arms * fall * 1.1 + core) * tex_;
                a *= sstep(0.025, 0.075, r) * sstep(1.0, 0.75, r);
            }
            d[y * stride + x] = (unsigned char)(255 * clamp01(a));
        }
    cairo_surface_mark_dirty(storm_spr);
}

/* --- per-frame --- */

static void update_climate(void)
{
    int step = (int)lround(wx.climate * CLIMATE_STEPS);
    if (step == climate_step)
        return;
    climate_step = step;
    recolor(step / (double)CLIMATE_STEPS);
    build_bg();
}

/*
 * Cloud cover (CPU load) sets the noise threshold, using the noise's own histogram
 * so cover really is the fraction of sky that's cloudy; density (RAM) sets opacity.
 */
static void update_cloud_lut(void)
{
    double cover = 0.08 + 0.44 * sstep(0, 1, wx.load);
    double dens = 0.55 + 0.42 * wx.ram;
    int total = cloud_cdf[255], thr = 255;
    for (int i = 0; i < 256; i++)
        if (cloud_cdf[i] >= total * (1 - cover)) {
            thr = i;
            break;
        }
    for (int i = 0; i < 256; i++)
        cloud_lut[i] = (uint8_t)(255 * dens * sstep(thr - 4, thr + 44 - 12 * wx.load, i));
}

/*
 * The globe, per pixel: surface (bilinear along longitude so the slow turn is smooth),
 * lit by the sun with a glint on the oceans, two cloud layers drifting at their own
 * speeds, rain clouds and lightning over the continents, then atmospheric haze.
 */
static void project(const double v[3], double k, double *x, double *y);
typedef struct { int x, y, r2, k; } spot;
static spot spots[MAX_BOLTS];
static int n_spots;

/* Lightning bolt brightness over its short life: flash, dim, re-strike, fade */
static double bolt_flicker(const bolt *b)
{
    double u = b->age / b->life;
    return u < 0.2 ? 1 : u < 0.35 ? 0.35 : u < 0.55 ? 0.9 : 0.9 * (1 - u) / 0.45;
}

/* Where lightning lights up the clouds this frame */
static void find_spots(void)
{
    n_spots = 0;
    for (int i = 0; i < MAX_BOLTS; i++) {
        const bolt *b = &bolts[i];
        double v[3], x, y;
        if (!b->alive)
            continue;
        to_view(b->lat, b->lon + (b->cloud ? theta_a : theta_s), v);
        if (v[2] < 0.05)
            continue;
        project(v, 1.0, &x, &y);
        int r = b->cloud ? 22 : 30;
        spots[n_spots++] = (spot){ (int)x, (int)y, r * r, (int)(230 * bolt_flicker(b) * sstep(0.05, 0.3, v[2])) };
    }
}

static void draw_globe(uint32_t *fb)
{
    const uint32_t full = (uint32_t)TW << 16;
    uint32_t rs = (uint32_t)(fmod(theta_s / (2 * M_PI), 1.0) * full);
    uint32_t ra = (uint32_t)(fmod(theta_a / (2 * M_PI), 1.0) * full);
    uint32_t rb = (uint32_t)(fmod(theta_b / (2 * M_PI), 1.0) * full);
    int rain[4] = { 0 };
    for (int d = 0; d < SYS_N_NVME; d++)
        rain[d + 1] = (int)(255 * 0.95 * sstep(0.03, 0.7, wx.disk[d]));
    /* clouds turn from white to slate as they get heavier */
    const double warm = sstep(0.5, 1.0, wx.climate);        /* dusty, cream clouds on a hot planet */
    const int cw_r = 244 - (int)(30 * wx.load) + (int)(6 * warm), cw_g = 246 - (int)(28 * wx.load) - (int)(8 * warm),
              cw_b = 252 - (int)(22 * wx.load) - (int)(30 * warm);
    find_spots();
    const int ar = atm_rgb[0], ag = atm_rgb[1], ab = atm_rgb[2];
    const int sr = storm_rgb[0], sg = storm_rgb[1], sb = storm_rgb[2];
    /* local, non-aliasing views of the maps so the compiler keeps them in registers */
    const uint32_t *restrict tx = tex;
    const uint8_t *restrict ca_map = cloud_a, *restrict cb_map = cloud_b, *restrict rn_map = rain_map;
    uint8_t lut[256];
    memcpy(lut, cloud_lut, sizeof(lut));
    uint8_t *restrict cn = cloud_n;
    const gpx *restrict gps = gp;
    uint32_t *restrict out = fb;
    const int ngp = n_gp;

    for (int i = 0; i < ngp; i++) {
        const gpx *p = &gps[i];
        uint32_t us = p->u - rs;
        uint32_t ui = (us >> 16) & TWM, f = (us >> 8) & 255, uj = (ui + 1) & TWM;
        uint32_t c0 = tx[p->row + ui], c1 = tx[p->row + uj];
        int r  = ((c0 >> 16 & 255) * (256 - f) + (c1 >> 16 & 255) * f) >> 8;
        int g  = ((c0 >> 8 & 255) * (256 - f) + (c1 >> 8 & 255) * f) >> 8;
        int b  = ((c0 & 255) * (256 - f) + (c1 & 255) * f) >> 8;
        int oc = ((c0 >> 24) * (256 - f) + (c1 >> 24) * f) >> 8;
        /* sunlight, less the shadow of the cloud layer sampled a few texels toward the sun */
        int shd = lut[ca_map[p->row + (((p->u - ra + (3u << 16)) >> 16) & TWM)]];
        int sh = p->shade * (256 - (shd * 80 >> 8)) >> 8;
        r = r * sh >> 8, g = g * sh >> 8, b = b * sh >> 8;
        int sp = p->spec * oc >> 8;
        r += sp, g += sp * 245 >> 8, b += sp * 225 >> 8;

        /* clouds */
        int na = ca_map[p->row + (((p->u - ra) >> 16) & TWM)];
        int nb = cb_map[p->row + (((p->u - rb) >> 16) & TWM)];
        int n = cloud_mix(na, nb);
        int ca = lut[n] * (180 + (nb * 76 >> 8)) >> 8;      /* fine texture inside the cloud */
        int cr = cw_r, cg = cw_g, cb = cw_b;
        int rm = rn_map[p->row + ui];
        if (rm) {
            int id = rm >> 6, w = (rm & 63) << 2;
            int ra_ = w * rain[id] >> 8;
            if (ra_) {
                int pop = (nb - 70) * 3;                    /* clumpy thunderheads, not a flat blob */
                pop = pop < 0 ? 0 : pop > 255 ? 255 : pop;
                ra_ = ra_ * (40 + (pop * 216 >> 8)) >> 8;
                int m = ra_ > ca ? ra_ : ca;
                int t = ra_ * 255 / (m + 1);                /* how much of it is rain cloud */
                cr += (sr - cr) * t >> 8, cg += (sg - cg) * t >> 8, cb += (sb - cb) * t >> 8;
                ca = m;
            }
        }
        int cs = p->cshade;
        cr = cr * cs >> 8, cg = cg * cs >> 8, cb = cb * cs >> 8;
        cn[p->off] = (uint8_t)ca;                      /* for the lightning pass: only clouds light up */
        r += (cr - r) * ca >> 8, g += (cg - g) * ca >> 8, b += (cb - b) * ca >> 8;

        /* atmosphere */
        int h = p->haze, hl = p->hlight;
        r += ((ar * hl >> 8) - r) * h >> 8, g += ((ag * hl >> 8) - g) * h >> 8, b += ((ab * hl >> 8) - b) * h >> 8;
        r = r > 255 ? 255 : r < 0 ? 0 : r;
        g = g > 255 ? 255 : g < 0 ? 0 : g;
        b = b > 255 ? 255 : b < 0 ? 0 : b;

        uint32_t *dst = &out[p->off];
        if (p->cover < 255) {
            int cv = p->cover;
            uint32_t o = *dst;
            r = ((int)(o >> 16 & 255) * (255 - cv) + r * cv) / 255;
            g = ((int)(o >> 8 & 255) * (255 - cv) + g * cv) / 255;
            b = ((int)(o & 255) * (255 - cv) + b * cv) / 255;
        }
        *dst = 0xff000000u | (uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b;
    }

    /* lightning lighting the clouds from inside, only around each strike */
    for (int k = 0; k < n_spots; k++) {
        const spot *sp = &spots[k];
        int r = (int)sqrt(sp->r2);
        for (int y = sp->y - r; y <= sp->y + r; y++)
            for (int x = sp->x - r; x <= sp->x + r; x++) {
                int dx = x - sp->x, dy = y - sp->y, d2 = dx * dx + dy * dy;
                int ex = x - CX, ey = y - CY;
                if (d2 >= sp->r2 || ex * ex + ey * ey >= (PR - 1) * (PR - 1))
                    continue;
                uint32_t *dst = &fb[y * fb_stride4 + x];
                int f = (sp->r2 - d2) * 256 / sp->r2;          /* soft falloff, no visible disc */
                int l = sp->k * (f * f >> 8) >> 8;
                l = l * (24 + cloud_n[y * fb_stride4 + x]) / 280;
                int cr = (*dst >> 16 & 255) + l, cg = (*dst >> 8 & 255) + l + (l >> 3), cb = (*dst & 255) + l + (l >> 2);
                *dst = 0xff000000u | (uint32_t)(cr > 255 ? 255 : cr) << 16 | (uint32_t)(cg > 255 ? 255 : cg) << 8 |
                       (uint32_t)(cb > 255 ? 255 : cb);
            }
    }
}

static void project(const double v[3], double k, double *x, double *y)
{
    *x = CX + PR * k * v[0];
    *y = CY - PR * k * v[1];
}

/*
 * Aurora: a curtain of light standing on the horizon above each pole, seen edge-on
 * along the limb like the photos from orbit. A band of radial rays follows the
 * planet's curve around the pole's direction on screen: every ray fades in from
 * nothing at its foot, peaks just above the horizon and fades out at the top, and
 * the band fades to nothing at both ends, so there are no hard edges anywhere.
 * Rays shimmer and drift sideways, the curtain's height waves slowly and the tops
 * lean in the wind. One cairo mesh pattern per pole (smooth colour and alpha across
 * every patch), drawn additively after the globe.
 */
static void draw_aurora(cairo_t *cr, int pole, double b, double t)
{
    const double D = M_PI / 180;
    rgb base = pole == 0 ? (rgb){ 0.40, 1.0, 0.62 } : (rgb){ 1.0, 0.42, 0.66 };
    rgb mid  = pole == 0 ? (rgb){ 0.12, 0.92, 0.72 } : (rgb){ 1.0, 0.36, 0.50 };
    rgb top  = pole == 0 ? (rgb){ 0.40, 0.42, 1.00 } : (rgb){ 1.0, 0.64, 0.30 };
    static const double stop_s[5] = { 0, 0.16, 0.38, 0.66, 1.0 };      /* foot .. top */
    static const double shape[5]  = { 0, 1.0,  0.50, 0.14, 0 };
    const rgb stop_c[5] = { base, base, mid, top, top };
    double v[3], ph = pole * 2.7;
    double bx[AUR_N + 1], by[AUR_N + 1], tx[AUR_N + 1], ty[AUR_N + 1], in[AUR_N + 1];

    /* where the pole sits on screen: the curtain is centred on that direction */
    to_view(pole == 0 ? M_PI / 2 : -M_PI / 2, 0, v);
    double pa = atan2(-v[1], v[0]);                     /* screen angle, y down */
    double half = (30 + 6 * b) * D;                     /* half the arc it spans */
    for (int i = 0; i <= AUR_N; i++) {
        double u = (double)i / AUR_N, th = pa + (2 * u - 1) * half;
        double env = pow(sin(M_PI * u), 1.6);           /* fades out at both ends */
        double wave = sin(th * 7 + t * 0.35 + ph) * 0.5 + sin(th * 13 - t * 0.22 + ph) * 0.5;
        double r0 = PR * (0.965 + 0.008 * wave);
        double hgt = PR * (0.07 + (0.06 + 0.30 * b) * (0.6 + 0.4 * sin(th * 5 - t * 0.3 + ph)) * (0.75 + 0.25 * env));
        double lean = 0.05 * sin(t * 0.4 + u * 5 + ph) + 0.03 * sin(t * 0.9 + u * 11);
        hgt *= 0.8 + 0.3 * sin(th * 31 + t * 0.45 + ph) * sin(th * 11 - t * 0.3);   /* ragged tops */
        bx[i] = CX + cos(th) * r0, by[i] = CY + sin(th) * r0;
        tx[i] = CX + cos(th + lean) * (r0 + hgt), ty[i] = CY + sin(th + lean) * (r0 + hgt);
        /* rays: soft streaks that drift along the curtain and flicker */
        double r1 = 0.5 + 0.5 * sin(th * 70 + t * 0.6 + 3.5 * sin(th * 6 + t * 0.2) + 1.5 * sin(th * 23 - t * 0.5));
        double r2 = 0.5 + 0.5 * sin(th * 113 - t * 1.1 + ph);
        double fold = 0.5 + 0.5 * sin(th * 17 + t * (0.3 + 0.5 * b) + ph);
        in[i] = 1.45 * b * env * (0.30 + 0.45 * pow(r1, 1.5) + 0.25 * r2) * (0.6 + 0.4 * fold);
    }

    cairo_pattern_t *mesh = cairo_pattern_create_mesh();
    int patches = 0;
    for (int i = 0; i < AUR_N; i++) {
        if (in[i] < 0.004 && in[i + 1] < 0.004)
            continue;
        for (int k = 0; k < 4; k++) {
            double px[4], py[4];
            const int ce[4] = { 0, 1, 1, 0 }, cs[4] = { 0, 0, 1, 1 };
            cairo_mesh_pattern_begin_patch(mesh);
            for (int c = 0; c < 4; c++) {
                int e = i + ce[c];
                double sv = stop_s[k + cs[c]];
                px[c] = bx[e] + (tx[e] - bx[e]) * sv;
                py[c] = by[e] + (ty[e] - by[e]) * sv;
                if (c == 0)
                    cairo_mesh_pattern_move_to(mesh, px[c], py[c]);
                else
                    cairo_mesh_pattern_line_to(mesh, px[c], py[c]);
            }
            for (int c = 0; c < 4; c++) {
                rgb cc = stop_c[k + cs[c]];
                cairo_mesh_pattern_set_corner_color_rgba(mesh, c, cc.r, cc.g, cc.b,
                                                         fmin(1, in[i + ce[c]] * shape[k + cs[c]]));
            }
            cairo_mesh_pattern_end_patch(mesh);
            patches++;
        }
    }
    if (patches) {
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        cairo_set_source(cr, mesh);
        cairo_paint(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    }
    cairo_pattern_destroy(mesh);
}

/* Storm cells: one per busy thread, riding the cloud layer, foreshortened on the sphere */
static void draw_storms(cairo_t *cr, double t)
{
    (void)t;
    for (int i = 0; i < n_threads; i++) {
        const storm *s = &storms[i];
        if (s->amp < 0.02)
            continue;
        double v[3], x, y;
        to_view(s->lat, s->lon + theta_a, v);
        if (v[2] < 0.03)
            continue;
        project(v, 1.0, &x, &y);
        double size = PR * (0.09 + 0.19 * s->amp);
        double lit = 0.05 + 0.95 * light_at(v, -0.16, 0.5);
        double phi = atan2(y - CY, x - CX);
        double hemi = s->lat >= 0 ? 1 : -1;
        cairo_save(cr);
        cairo_translate(cr, x, y);
        cairo_rotate(cr, phi);
        cairo_scale(cr, v[2], 1);
        cairo_rotate(cr, -phi);
        cairo_rotate(cr, s->spin);
        cairo_scale(cr, size / (STORM_SPR / 2.0), hemi * size / (STORM_SPR / 2.0));
        double a = fmin(1, s->amp * 2.0) * sstep(0.03, 0.3, v[2]);
        cairo_pattern_t *mask = cairo_pattern_create_for_surface(storm_spr);
        cairo_pattern_set_filter(mask, CAIRO_FILTER_BILINEAR);
        cairo_matrix_t mm;
        cairo_matrix_init_translate(&mm, STORM_SPR / 2.0 - 5, STORM_SPR / 2.0 - 7);
        cairo_pattern_set_matrix(mask, &mm);
        cairo_set_source_rgba(cr, 0, 0.01, 0.03, 0.45 * a);        /* shadow on the cloud deck below */
        cairo_mask(cr, mask);
        cairo_matrix_init_translate(&mm, STORM_SPR / 2.0, STORM_SPR / 2.0);
        cairo_pattern_set_matrix(mask, &mm);
        cairo_set_source_rgba(cr, 0.96 * lit, 0.97 * lit, 1.0 * lit, a * (0.75 + 0.25 * wx.ram));
        cairo_mask(cr, mask);
        cairo_pattern_destroy(mask);
        cairo_restore(cr);
    }
}

/* Jet-stream streaks just above the clouds: visible in front, or peeking round the limb */
static int jet_point(const jet *j, double lon, double *x, double *y)
{
    double v[3], k = 1.02;
    to_view(j->lat + 0.035 * sin(lon * 3 + j->wob), lon, v);
    project(v, k, x, y);
    return v[2] > 0 || k * k * (v[0] * v[0] + v[1] * v[1]) > 1.0;
}

static void draw_jets(cairo_t *cr)
{
    const int K = 7;
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < N_JETS; i++) {
        const jet *j = &jets[i];
        if (!j->alive)
            continue;
        double env = sstep(0, 1.0, j->age) * sstep(j->life, j->life - 1.5, j->age);
        double act = j->dir > 0 ? wx.tx : wx.rx;
        double a = env * (0.16 + 0.30 * act);
        if (a < 0.01)
            continue;
        double hx = 0, hy = 0, ex = 0, ey = 0, hv[3];
        int pen = 0, any = 0;
        to_view(j->lat, j->lon, hv);
        a *= 0.15 + 0.85 * pow(1 - fabs(hv[2]), 2);    /* faint over the face, bright wrapping the limb */
        cairo_new_path(cr);
        for (int k = 0; k <= K; k++) {
            double x, y, lon = j->lon - j->dir * j->len * k / K;
            int vis = jet_point(j, lon, &x, &y);
            if (k == 0)
                hx = x, hy = y;
            if (vis) {
                if (pen)
                    cairo_line_to(cr, x, y);
                else
                    cairo_move_to(cr, x, y);
                pen = 1, any = 1, ex = x, ey = y;
            } else
                pen = 0;
        }
        if (!any)
            continue;
        rgb c = j->dir > 0 ? (rgb){ 1.0, 0.82, 0.52 } : (rgb){ 0.55, 0.85, 1.0 };
        cairo_pattern_t *pat = cairo_pattern_create_linear(hx, hy, ex, ey);
        cairo_pattern_add_color_stop_rgba(pat, 0, c.r, c.g, c.b, a);
        cairo_pattern_add_color_stop_rgba(pat, 1, c.r, c.g, c.b, 0);
        cairo_set_source(cr, pat);
        cairo_set_line_width(cr, 1.0 + 0.6 * act);
        cairo_stroke(cr);
        cairo_pattern_destroy(pat);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

/* Lightning: a flash of light in the cloud and, briefly, a forked bolt */
static void draw_bolts(cairo_t *cr)
{
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    for (int i = 0; i < MAX_BOLTS; i++) {
        const bolt *b = &bolts[i];
        if (!b->alive)
            continue;
        double v[3], x, y;
        to_view(b->lat, b->lon + (b->cloud ? theta_a : theta_s), v);
        if (v[2] < 0.12)
            continue;
        project(v, 1.0, &x, &y);
        double k = bolt_flicker(b) * sstep(0.12, 0.35, v[2]);
        double gr = 9 + 7 * k;
        cairo_pattern_t *pat = cairo_pattern_create_radial(x, y, 0, x, y, gr);
        cairo_pattern_add_color_stop_rgba(pat, 0, 0.85, 0.90, 1.0, 0.6 * k);
        cairo_pattern_add_color_stop_rgba(pat, 0.3, 0.55, 0.65, 1.0, 0.30 * k);
        cairo_pattern_add_color_stop_rgba(pat, 1, 0.4, 0.5, 1.0, 0);
        cairo_set_source(cr, pat);
        cairo_arc(cr, x, y, gr, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(pat);

        /* the bolt itself: a jagged fork, foreshortened like the ground under it */
        unsigned seed = b->seed;
        double px = x, py = y, ang = b->ang, bx[8], by[8];
        int nseg = 6;
        bx[0] = px, by[0] = py;
        for (int s = 1; s <= nseg; s++) {
            ang += ((rand_r(&seed) % 1000) / 1000.0 - 0.5) * 1.3;
            double len = 3.2 + (rand_r(&seed) % 1000) / 1000.0 * 3.5;
            px += cos(ang) * len * (0.4 + 0.6 * v[2]);
            py += sin(ang) * len;
            bx[s] = px, by[s] = py;
        }
        for (int pass = 0; pass < 2; pass++) {
            cairo_move_to(cr, bx[0], by[0]);
            for (int s = 1; s <= nseg; s++)
                cairo_line_to(cr, bx[s], by[s]);
            int fs = 2 + rand_r(&seed) % 3;                    /* a side branch */
            cairo_move_to(cr, bx[fs], by[fs]);
            cairo_line_to(cr, bx[fs] + cos(b->ang + 1.1) * 7, by[fs] + sin(b->ang + 1.1) * 7);
            cairo_set_line_width(cr, pass ? 1.1 : 3.5);
            cairo_set_source_rgba(cr, 0.8, 0.88, 1.0, (pass ? 0.95 : 0.28) * k);
            cairo_stroke(cr);
        }
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

/* Meteors: a bright head and a fading trail, flaring as they burn up */
static void draw_meteors(cairo_t *cr)
{
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < MAX_METEORS; i++) {
        const meteor *m = &meteors[i];
        if (!m->alive)
            continue;
        double u = m->age / m->life;
        double k = sstep(0, 0.3, u) * (1 + 1.2 * sstep(0.7, 0.95, u)) * sstep(1.0, 0.94, u);
        double sp = hypot(m->vx, m->vy), len = 18 + 34 * sstep(0, 0.6, u);
        double tx = m->x - m->vx / sp * len, ty = m->y - m->vy / sp * len;
        cairo_pattern_t *pat = cairo_pattern_create_linear(m->x, m->y, tx, ty);
        cairo_pattern_add_color_stop_rgba(pat, 0, 0.90, 1.0, 0.88, 0.85 * fmin(1, k));
        cairo_pattern_add_color_stop_rgba(pat, 0.3, 0.55, 0.95, 0.75, 0.35 * fmin(1, k));
        cairo_pattern_add_color_stop_rgba(pat, 1, 0.4, 0.7, 1.0, 0);
        cairo_set_source(cr, pat);
        cairo_set_line_width(cr, 1.4 + 0.6 * k);
        cairo_move_to(cr, m->x, m->y);
        cairo_line_to(cr, tx, ty);
        cairo_stroke(cr);
        cairo_pattern_destroy(pat);
        double gr = 3 + 3 * k;
        pat = cairo_pattern_create_radial(m->x, m->y, 0, m->x, m->y, gr);
        cairo_pattern_add_color_stop_rgba(pat, 0, 1, 1, 0.92, 0.9 * fmin(1, k));
        cairo_pattern_add_color_stop_rgba(pat, 1, 0.6, 1, 0.8, 0);
        cairo_set_source(cr, pat);
        cairo_arc(cr, m->x, m->y, gr, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(pat);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

/* --- simulation --- */

static double disk_act(double bytes)  { return clamp01(log10(bytes / 1e6 + 1) / log10(3001)); }   /* 1 MB/s .. 3 GB/s */
static double net_act(double bytes)   { return clamp01(log10(bytes / 1e4 + 1) / log10(10001)); }  /* 10 kB/s .. 100 MB/s */

static void ease(double *v, double target, double dt, double tau)
{
    *v += (target - *v) * (1 - exp(-dt / tau));
}

static void simulate(const sys_stats *s, const stats *g, double dt, double t)
{
    const double D = M_PI / 180;
    double temp = isnan(s->cpu_temp) ? 50 : s->cpu_temp;
    ease(&wx.temp, temp, dt, 2.0);
    ease(&wx.climate, clamp01((wx.temp - TEMP_COOL) / (TEMP_HOT - TEMP_COOL)), dt, 1.0);
    ease(&wx.load, s->cpu_total, dt, 1.2);
    ease(&wx.ram, s->ram_used, dt, 2.0);
    for (int d = 0; d < SYS_N_NVME; d++)
        ease(&wx.disk[d], disk_act(s->nvme_read[d] + s->nvme_write[d]), dt, 0.9);
    ease(&wx.rx, net_act(s->net_rx), dt, 1.0);
    ease(&wx.tx, net_act(s->net_tx), dt, 1.0);
    for (int i = 0; i < N_GPUS; i++) {
        /* power, or the GPU-mode activity figure; vLLM tokens count too when serving */
        double a = clamp01((g->power[i] - 30) / 470.0);
        if (gpu_source || g->tok_port[i] > 0)
            a = fmax(a, clamp01(g->tok_port[i] / GPU_FULL_RATE));
        ease(&wx.aur[i], 0.10 + 0.90 * a, dt, 0.8);
    }
    ease(&wx.clock, clamp01((s->cpu_mhz_avg - 1500) / 4000), dt, 1.5);
    ease(&wx.intr, clamp01((log10(s->intr_s + 1) - 4.3) / 2.2), dt, 1.0);
    ease(&wx.meteors, 0.6 * pow(fmax(0, log10(s->pgmajfault_s + 1) - 2.5), 1.5), dt, 1.0);
    double watts = isnan(s->cpu_watts) ? 35 + 150 * s->cpu_total : s->cpu_watts;   /* estimate without root */
    ease(&wx.sun, clamp01((watts - 30) / 170), dt, 1.0);
    ease(&wx.psi, fmax(s->psi_cpu, fmax(s->psi_io, s->psi_mem)), dt, 2.0);
    pressure_hpa = 1022 - 0.9 * wx.psi;
    uptime_days = s->uptime_s / 86400;
    n_threads = s->n_cpus > 0 ? s->n_cpus : 32;
    for (int i = 0; i < n_threads; i++)
        ease(&wx.cpu[i], s->cpu[i], dt, 1.0);

    update_climate();
    update_cloud_lut();

    /* Rotation: the surface turns once per DAY_S, clouds drift a little faster, more in wind */
    double wind = 0.5 * (wx.rx + wx.tx);
    theta_s += dt * 2 * M_PI / DAY_S;
    theta_a += dt * (2 * M_PI / DAY_S + 0.004 + 0.030 * wx.clock + 0.015 * wind);
    theta_b += dt * (2 * M_PI / DAY_S + 0.012 + 0.050 * wx.clock + 0.030 * wind);

    /* Storms: the busiest threads (above STORM_MIN_LOAD), at most MAX_STORMS */
    int rank[SYS_MAX_CPUS];
    for (int i = 0; i < n_threads; i++)
        rank[i] = i;
    for (int i = 1; i < n_threads; i++)                    /* insertion sort, busiest first */
        for (int j = i; j > 0 && wx.cpu[rank[j]] > wx.cpu[rank[j - 1]]; j--) {
            int tmp = rank[j];
            rank[j] = rank[j - 1];
            rank[j - 1] = tmp;
        }
    for (int r = 0; r < n_threads; r++) {
        storm *st = &storms[rank[r]];
        double target = r < MAX_STORMS ? sstep(STORM_MIN_LOAD, 0.95, wx.cpu[rank[r]]) : 0;
        ease(&st->amp, target, dt, 0.9);
        st->spin += dt * (0.25 + 0.9 * st->amp) * (0.5 + wx.clock) * (st->lat >= 0 ? -1 : 1);
    }

    /* Lightning inside the storm cells, as often as the interrupts fire */
    double amp_sum = 0;
    for (int i = 0; i < n_threads; i++)
        amp_sum += storms[i].amp > 0.05 ? storms[i].amp : 0;
    if (amp_sum > 0.05 && wx.intr > 0.05) {
        storm_acc += dt * wx.intr * (0.4 + 1.3 * fmin(amp_sum, 4)) * (0.6 + 0.8 * frand());
        while (storm_acc >= 1) {
            storm_acc -= 1;
            double pick = frand() * amp_sum;
            for (int i = 0; i < n_threads; i++) {
                const storm *st = &storms[i];
                if (st->amp <= 0.05 || (pick -= st->amp) > 0)
                    continue;
                for (int k = 0; k < MAX_BOLTS; k++)
                    if (!bolts[k].alive) {
                        double a = frand() * 2 * M_PI, r = (0.02 + 0.05 * frand()) * (0.5 + st->amp);
                        bolts[k] = (bolt){ st->lat + sin(a) * r, st->lon + cos(a) * r / cos(st->lat), 0,
                                           0.22 + 0.12 * frand(), frand() * 2 * M_PI, (unsigned)rand(), 1, 1 };
                        break;
                    }
                break;
            }
        }
    }

    /* Meteors: major page faults burning up in the atmosphere */
    meteor_acc += dt * wx.meteors;
    while (meteor_acc >= 1) {
        meteor_acc -= 1;
        for (int i = 0; i < MAX_METEORS; i++)
            if (!meteors[i].alive) {
                double a = frand() * 2 * M_PI;
                if (sin(a) > 0.5)
                    a = -a;                             /* keep clear of the text at the bottom */
                double sx = CX + cos(a) * PR * 1.45, sy = CY + sin(a) * PR * 1.45;
                double ta = a + (frand() < 0.5 ? 1 : -1) * (0.25 + 0.45 * frand());
                double ex = CX + cos(ta) * PR * 1.01, ey = CY + sin(ta) * PR * 1.01;
                double dist = hypot(ex - sx, ey - sy), spd = 220 + 140 * frand();
                meteors[i] = (meteor){ sx, sy, (ex - sx) / dist * spd, (ey - sy) / dist * spd, 0,
                                       dist / spd * (0.85 + 0.2 * frand()), 1 };
                break;
            }
    }
    for (int i = 0; i < MAX_METEORS; i++) {
        meteor *m = &meteors[i];
        if (!m->alive)
            continue;
        m->x += m->vx * dt, m->y += m->vy * dt;
        if ((m->age += dt) > m->life)
            m->alive = 0;
    }

    /* Lightning: strikes per second grow with each drive's I/O */
    for (int d = 0; d < SYS_N_NVME; d++) {
        double act = wx.disk[d];
        if (act < 0.06)
            continue;
        strike_acc[d] += dt * (0.3 + 5.5 * pow(act, 1.5)) * (0.6 + 0.8 * frand());
        while (strike_acc[d] >= 1) {
            strike_acc[d] -= 1;
            for (int i = 0; i < MAX_BOLTS; i++)
                if (!bolts[i].alive) {
                    double a = frand() * 2 * M_PI, r = sqrt(frand()) * cont_rad[d] * 0.7;
                    bolts[i] = (bolt){ (cont_lat[d] + sin(a) * r) * D, (cont_lon[d] + cos(a) * r / cos(cont_lat[d] * D)) * D,
                                       0, 0.30 + 0.15 * frand(), frand() * 2 * M_PI, (unsigned)rand(), 1, 0 };
                    break;
                }
        }
    }
    for (int i = 0; i < MAX_BOLTS; i++)
        if (bolts[i].alive && (bolts[i].age += dt) > bolts[i].life)
            bolts[i].alive = 0;

    /* Wind: jet streaks, download (in) blowing west, upload (out) blowing east */
    int count[2] = { 0, 0 };
    for (int i = 0; i < N_JETS; i++) {
        jet *j = &jets[i];
        if (!j->alive)
            continue;
        double act = j->dir > 0 ? wx.tx : wx.rx;
        j->lon += j->dir * j->spd * (0.10 + 1.1 * act) * dt;
        j->len = 0.12 + 0.55 * act;
        if ((j->age += dt) > j->life)
            j->alive = 0;
        else
            count[j->dir > 0]++;
    }
    for (int dir = 0; dir < 2; dir++) {
        double act = dir ? wx.tx : wx.rx;
        int want = (int)(2 + 22 * act);
        if (count[dir] < want && frand() < dt * 8) {
            for (int i = 0; i < N_JETS; i++)
                if (!jets[i].alive) {
                    double lat = (14 + frand() * 46) * (frand() < 0.5 ? 1 : -1);
                    jets[i] = (jet){ lat * D, frand() * 2 * M_PI, 0.8 + 0.4 * frand(), 0, 4 + 4 * frand(),
                                     0.12, frand() * 6, dir ? 1 : -1, 1 };
                    break;
                }
        }
    }
    (void)t;
}

/* --- HUD --- */

static const char *FONT = "Fira Sans";

static double text_w(cairo_t *cr, const char *s, double size, double track, int bold)
{
    cairo_text_extents_t e;
    cairo_select_font_face(cr, FONT, CAIRO_FONT_SLANT_NORMAL, bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &e);
    return e.x_advance + track * (strlen(s) - 1);
}

/* Text with letter spacing, a soft dark halo behind it for legibility over stars */
static void track_text(cairo_t *cr, const char *s, double x, double y, double size, double track, int bold, rgb c, double alpha)
{
    cairo_select_font_face(cr, FONT, CAIRO_FONT_SLANT_NORMAL, bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    double px = x;
    cairo_new_path(cr);
    for (const char *p = s; *p; ) {
        char ch[8];
        int n = (*p & 0x80) ? ((*p & 0xE0) == 0xC0 ? 2 : 3) : 1;   /* whole UTF-8 characters */
        memcpy(ch, p, n);
        ch[n] = 0;
        cairo_text_extents_t e;
        cairo_text_extents(cr, ch, &e);
        cairo_move_to(cr, px, y);
        cairo_text_path(cr, ch);
        px += e.x_advance + track;
        p += n;
    }
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, size * 0.18);
    cairo_set_source_rgba(cr, 0.0, 0.01, 0.04, 0.55 * alpha);
    cairo_stroke_preserve(cr);
    cairo_set_source_rgba(cr, c.r, c.g, c.b, alpha);
    cairo_fill(cr);
}

enum { W_CLEAR, W_FAIR, W_CLOUDY, W_WINDY, W_RAIN, W_THUNDER, W_STORMY, W_SEVERE, W_HEATWAVE };
static const char *W_NAME[] = { "CLEAR", "FAIR", "CLOUDY", "WINDY", "RAIN", "THUNDER", "STORMY", "SEVERE", "HEATWAVE" };

/*
 * The forecast word. Each condition has an enter and a lower leave threshold, and the
 * shown word must be wrong for 2 s before it changes, so the word never flickers.
 */
static int forecast(double t)
{
    static int shown = W_CLEAR, pending = -1;
    static double since;
    double disk = fmax(wx.disk[0], fmax(wx.disk[1], wx.disk[2])), wind = fmax(wx.rx, wx.tx);
    double hy = 0.04;                                   /* hysteresis */
#define ON(cond_enter, cond_stay, w) ((shown == (w)) ? (cond_stay) : (cond_enter))
    int w;
    if (ON(wx.temp >= 88, wx.temp >= 84, W_HEATWAVE))
        w = W_HEATWAVE;
    else if (ON(wx.load > 0.80, wx.load > 0.80 - hy, W_SEVERE))
        w = W_SEVERE;
    else if (ON(disk > 0.45, disk > 0.45 - hy, W_THUNDER))
        w = W_THUNDER;
    else if (ON(wx.load > 0.50, wx.load > 0.50 - hy, W_STORMY))
        w = W_STORMY;
    else if (ON(disk > 0.15, disk > 0.15 - hy, W_RAIN))
        w = W_RAIN;
    else if (ON(wind > 0.55, wind > 0.55 - hy, W_WINDY))
        w = W_WINDY;
    else if (ON(wx.load > 0.25, wx.load > 0.25 - hy, W_CLOUDY))
        w = W_CLOUDY;
    else if (ON(wx.load > 0.08, wx.load > 0.08 - hy / 2, W_FAIR))
        w = W_FAIR;
    else
        w = W_CLEAR;
#undef ON
    /* the timer runs from when the shown word stopped being right, whatever replaces it */
    if (w == shown)
        pending = -1;
    else {
        if (pending < 0)
            since = t;
        pending = w;
        if (t - since > 2.0)
            shown = w, pending = -1;
    }
    return shown;
}

/* Width of the round screen at height y, less a margin */
static double chord_at(double y, double margin)
{
    double dy = fabs(y - SIZE / 2.0);
    return dy >= SIZE / 2.0 ? 0 : 2 * sqrt(SIZE * SIZE / 4.0 - dy * dy) - 2 * margin;
}

static void update_hud(double t)
{
    static double next;
    static int temp_shown = -100, hpa_shown, word, baro;
    char key[128], num[16], line1[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        if (fabs(wx.temp - temp_shown) > 0.7)       /* hysteresis: no 62/63 flip-flop */
            temp_shown = (int)lround(wx.temp);
        if (fabs(pressure_hpa - hpa_shown) > 0.7)
            hpa_shown = (int)lround(pressure_hpa);
        /* barometer: HIGH / steady / LOW, each with a margin before it changes back */
        if (baro == 0)
            baro = hpa_shown >= 1016 ? 1 : hpa_shown <= 1006 ? -1 : 0;
        else if ((baro > 0 && hpa_shown < 1014) || (baro < 0 && hpa_shown > 1008))
            baro = 0;
        word = forecast(t);
    }
    snprintf(num, sizeof(num), "%d\xc2\xb0" "C", temp_shown);
    snprintf(line1, sizeof(line1), "DAY %d  \xc2\xb7  %d hPa%s", (int)uptime_days + 1, hpa_shown,
             baro > 0 ? " HIGH" : baro < 0 ? " LOW" : "");
    snprintf(key, sizeof(key), "%s|%s|%d|%d", num, line1, word, climate_step / 4);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);

    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* line 1: day and barometer, fitted to the chord at its baseline */
    double y1 = 411, s1 = 22, tr1 = 1.5;
    while (text_w(cr, line1, s1, tr1, 0) > chord_at(y1, 16) && tr1 > 0)
        tr1 -= 0.25;
    double w1 = text_w(cr, line1, s1, tr1, 0);
    track_text(cr, line1, SIZE / 2.0 - w1 / 2, y1 - HUD_Y, s1, tr1, 0, (rgb){ 0.70, 0.78, 0.90 }, 0.80);

    /* line 2: "62°C  STORMY", fitted to the chord at its baseline */
    const char *wtxt = W_NAME[word];
    double y2 = 447, size = 30, wsize = 24, track = 3, gap = 14, total;
    for (;;) {
        total = text_w(cr, num, size, 0, 1) + gap + text_w(cr, wtxt, wsize, track, 0);
        if (total <= chord_at(y2, 14) || size <= 24)
            break;
        size -= 1, wsize = fmax(22, wsize - 0.5), track = fmax(1, track - 0.3), gap = fmax(10, gap - 0.5);
    }
    double x = SIZE / 2.0 - total / 2;
    rgb tc = lerp((rgb){ 0.70, 0.88, 1.0 }, (rgb){ 1.0, 0.72, 0.42 }, sstep(0.3, 1.0, wx.climate));
    track_text(cr, num, x, y2 - HUD_Y, size, 0, 1, tc, 1.0);
    x += text_w(cr, num, size, 0, 1) + gap;
    track_text(cr, wtxt, x, y2 - HUD_Y, wsize, track, 0, (rgb){ 0.88, 0.92, 0.97 }, 0.92);
    cairo_destroy(cr);
}

/* --- frame --- */

static void render(cairo_surface_t *surf, cairo_t *cr, double t)
{
    uint32_t *fb = (uint32_t *)cairo_image_surface_get_data(surf);
    int stride = cairo_image_surface_get_stride(surf);
    cairo_surface_flush(surf);
    memcpy(fb, cairo_image_surface_get_data(bg_img), (size_t)stride * SIZE);
    cairo_surface_mark_dirty(surf);

    /* the sun's glare, as bright as the CPU package is drawing power */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_set_source_surface(cr, sun_img, 0, 0);
    cairo_rectangle(cr, 0, 0, SUN_W, SUN_W);
    cairo_clip(cr);
    cairo_paint_with_alpha(cr, 0.25 + 0.75 * wx.sun);
    cairo_reset_clip(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    cairo_surface_flush(surf);
    draw_globe(fb);
    cairo_surface_mark_dirty(surf);

    draw_storms(cr, t);
    draw_bolts(cr);
    draw_aurora(cr, 0, wx.aur[0], t);
    draw_aurora(cr, 1, wx.aur[1], t);
    draw_jets(cr);
    draw_meteors(cr);

    update_hud(t);
    cairo_set_source_surface(cr, hud_cache, 0, HUD_Y);
    cairo_paint(cr);
}

static void init_world(int stride4)
{
    view_matrix();
    build_world();
    build_globe(stride4);
    build_stars();
    build_storm_sprite();
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, HUD_H);
    srand(4242);
    for (int i = 0; i < SYS_MAX_CPUS; i++) {                /* each thread's storm has a home */
        double lat = (12 + frand() * 36) * (i % 2 ? -1 : 1);
        storms[i] = (storm){ lat * M_PI / 180, (i * 7 % 32) / 32.0 * 2 * M_PI + frand() * 0.2, 0, frand() * 6 };
    }
    wx.temp = TEMP_COOL;
    srand((unsigned)time(NULL));
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

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    init_world(cairo_image_surface_get_stride(surf) / 4);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* fixed scenes from the showcase script: maxed, building, calm */
        struct { double at; const char *png; } scenes[] = {
            { 24.0, "weather_preview.png" },
            { 13.0, "weather_mid.png" },
            { 2.0, "weather_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            size_t len = 0;
            double b0 = 0;
            int n = 8 * FPS_BUSY;
            srand(7 + k);
            hud_key[0] = 0;
            for (int i = 0; i < n; i++) {
                if (i == n - 60)
                    b0 = now_s();
                double t = scenes[k].at - 8 + i / (double)FPS_BUSY;
                showcase_poll(&s, &g, scenes[k].at);        /* hold the scene steady */
                simulate(&s, &g, 1.0 / FPS_BUSY, t);
                render(surf, cr, t);
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
            showcase_poll(&s, &g, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 0.5;
            if (demo) {
                /* fresh values each poll from a separate copy; never scale s in place */
                static sys_stats demo_sys;
                static stats demo_gpu;
                demo_poll(&demo_sys, &demo_gpu, t - t0);
                s = demo_sys;
                g = demo_gpu;
            } else {
                sys_poll(&s, t);
                gpus_poll(&g);
                if (gpu_source)
                    gpu_rate_poll(&g);
                else
                    vllm_poll(&g, t);
            }
        }

        simulate(&s, &g, dt, t - t0);
        render(surf, cr, t - t0);
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

        /* Calm weather runs at the idle frame rate */
        double disk = fmax(wx.disk[0], fmax(wx.disk[1], wx.disk[2]));
        int idle = !showcase && wx.load < 0.12 && disk < 0.1 && fmax(wx.rx, wx.tx) < 0.45 &&
                   fmax(wx.aur[0], wx.aur[1]) < 0.3;
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
    cairo_surface_destroy(stars_img);
    cairo_surface_destroy(bg_img);
    cairo_surface_destroy(storm_spr);
    cairo_surface_destroy(hud_cache);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
