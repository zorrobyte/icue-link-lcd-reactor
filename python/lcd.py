"""Minimal driver for the Corsair iCUE LINK AIO LCD Screen Module (1b1c:0c4e).

Frames are 480x480 JPEGs split into 1024-byte HID output reports:
  [0x02, 0x05, 0x01, last, chunk_index, 0x00, len_lo, len_hi] + up to 1016 bytes of JPEG
Protocol as implemented by OpenLinkHub (src/devices/lsh/lsh.go transferToLcd).
"""
import fcntl
import glob
import io
import os

REPORT_SIZE = 1024
HEADER_SIZE = 8
CHUNK_SIZE = REPORT_SIZE - HEADER_SIZE
WIDTH = HEIGHT = 480


def find_hidraw():
    for uevent in glob.glob("/sys/class/hidraw/hidraw*/device/uevent"):
        with open(uevent) as f:
            if "00001B1C:00000C4E" in f.read().upper():
                return "/dev/" + uevent.split("/")[4]
    raise FileNotFoundError("iCUE LINK LCD (1b1c:0c4e) not found")


def _hidiocsfeature(length):
    # _IOC(_IOC_WRITE|_IOC_READ, 'H', 0x06, len)
    return (3 << 30) | (length << 16) | (ord("H") << 8) | 0x06


class LinkLCD:
    def __init__(self, path=None):
        self.path = path or find_hidraw()
        self.fd = os.open(self.path, os.O_RDWR)

    def close(self):
        os.close(self.fd)

    def feature(self, data):
        buf = bytearray(data)
        fcntl.ioctl(self.fd, _hidiocsfeature(len(buf)), buf)

    def set_brightness(self, percent):
        self.feature([0x03, 0x0B, max(0, min(100, int(percent))), 0x01])

    def set_rotation(self, rotation):
        # 0..3 in 90 degree steps
        self.feature([0x03, 0x0C, rotation & 0x03, 0x01])

    def send_jpeg(self, jpeg):
        chunks = [jpeg[i:i + CHUNK_SIZE] for i in range(0, len(jpeg), CHUNK_SIZE)]
        for i, chunk in enumerate(chunks):
            report = bytearray(REPORT_SIZE)
            report[0:3] = b"\x02\x05\x01"
            report[3] = 0x01 if i == len(chunks) - 1 else 0x00
            report[4] = i & 0xFF
            report[6] = len(chunk) & 0xFF
            report[7] = (len(chunk) >> 8) & 0xFF
            report[HEADER_SIZE:HEADER_SIZE + len(chunk)] = chunk
            os.write(self.fd, report)

    def send_image(self, img, quality=90):
        buf = io.BytesIO()
        img.convert("RGB").resize((WIDTH, HEIGHT)).save(buf, "JPEG", quality=quality)
        self.send_jpeg(buf.getvalue())
