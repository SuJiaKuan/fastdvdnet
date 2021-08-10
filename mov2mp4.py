import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Application to convert mov videos to mp4 format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input folder",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output folder",
    )

    args = parser.parse_args()

    return args


def convert(in_path, out_path, codec_args, crf, cmdout):
    cmd = ["ffmpeg", "-y", "-i", in_path]
    cmd += codec_args
    cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-crf", crf, "-an", out_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, stdout=cmdout, stderr=cmdout)


def convert_scenes(
    datab_root,
    out_root,
    codec="h264",
    crf="18",
    keyint="4",
    quiet=False,
):
    assert codec in ['h264', 'hevc'], '--codec must be one of h264 or hevc'

    # Codec options
    codec_args = ["-preset", "slow"]
    if codec == 'h264':
        codec_args = ["-c:v", "libx264", "-g", keyint,
                      "-profile:v", "high"]
    elif codec == 'hevc' or codec == 'h265':
        codec_args = ["-c:v", "libx265", "-x265-params",
                      "keyint=%s:no-open-gop=1" % (keyint)]
    else:
        raise ValueError("Unknown codec")

    # Quiet mode
    if quiet:
        cmdout = subprocess.DEVNULL
    else:
        cmdout = None

    # Output dir
    if not os.path.isdir(out_root):
        os.makedirs(out_root)
    print('Writing sequences to {}'.format(out_root))

    for filename in os.listdir(datab_root):
        in_path = os.path.join(datab_root, filename)
        if not in_path.endswith(".mov"):
            continue

        out_path = os.path.join(out_root, filename.replace(".mov", ".mp4"))
        convert(in_path, out_path, codec_args, crf, cmdout)


def main(args):
    convert_scenes(args.input, args.output)


if __name__ == "__main__":
    main(parse_args())
