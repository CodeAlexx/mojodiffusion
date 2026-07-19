from ffmpeg.cli import have_ffmpeg, decode_to_wav, read_audio, mux_av, probe_to_file

def main() raises:
    if not have_ffmpeg():
        print("SKIP: ffmpeg not on PATH")
        return

    # 1) decode the mp3 (cry.wav is mp3-in-wav-ext) -> samples via ffmpeg+wav reader
    var buf = read_audio(String("/home/alex/Downloads/cry.wav"),
                         String("/tmp/cry_decoded.wav"))
    print("DECODE rate", buf.rate, "ch", buf.channels, "frames", buf.num_frames(),
          "secs", buf.duration_secs())
    var ok = buf.num_frames() > 0 and buf.rate > 0

    # 2) mux 4 PNG frames + a wav -> mp4 (frames prepared by the test driver)
    var rc = mux_av(String("/tmp/mf_%02d.png"), String("/tmp/sine16.wav"),
                    String("/tmp/mojo_av.mp4"), 2)
    print("MUX rc", rc)

    _ = probe_to_file(String("/tmp/mojo_av.mp4"), String("/tmp/mojo_av_probe.json"))
    print("ffmpeg test:", "OK" if (ok and rc == 0) else "FAIL")
    if not (ok and rc == 0):
        raise Error("ffmpeg test FAILED")
