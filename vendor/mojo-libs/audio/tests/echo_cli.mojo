# reads argv[1] wav, writes argv[2] as float32 wav (decode+encode round-trip via the lib)
from std.sys import argv
from audio.wav import read_wav, write_wav
def main() raises:
    var a = argv()
    var buf = read_wav(String(a[1]))
    print("mojo:", buf.rate, buf.channels, buf.num_frames())
    write_wav(String(a[2]), buf, 32)
