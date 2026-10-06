"""Compare a compiled solution with independent references and time large cases."""

import argparse
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import time


def normalize(value):
    if value < 0:
        return (-value) % 241
    return value % 251 if value > 255 else value


def pack_input(height, width, pixels, kh, kw, kernel):
    return (struct.pack("=II", height, width) + pixels
            + struct.pack("=HH", kh, kw)
            + bytes(value & 255 for value in kernel))


def reference_python(height, width, pixels, kh, kw, kernel):
    result = bytearray()
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                total = 0
                for ky in range(kh):
                    sy = y + ky - kh // 2
                    for kx in range(kw):
                        sx = x + kx - kw // 2
                        if 0 <= sy < height and 0 <= sx < width:
                            total += (pixels[(sy * width + sx) * 3 + channel]
                                      * kernel[ky * kw + kx])
                result.append(normalize(total))
    return struct.pack("=II", height, width) + result


def reference_numpy(np, height, width, pixels, kh, kw, kernel):
    source = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    source = source.astype(np.int64)
    sums = np.zeros((height, width, 3), dtype=np.int64)
    # Accumulate shifted whole arrays, rather than visiting output pixels.
    for ky in range(kh):
        dy = ky - kh // 2
        y0, y1 = max(0, -dy), min(height, height - dy)
        for kx in range(kw):
            dx = kx - kw // 2
            x0, x1 = max(0, -dx), min(width, width - dx)
            coefficient = kernel[ky * kw + kx]
            if coefficient and y0 < y1 and x0 < x1:
                sums[y0:y1, x0:x1] += (
                    source[y0 + dy:y1 + dy, x0 + dx:x1 + dx] * coefficient)
    result = np.where(sums < 0, (-sums) % 241,
                      np.where(sums > 255, sums % 251, sums))
    return struct.pack("=II", height, width) + result.astype(np.uint8).tobytes()


def reference_constant(height, width, kh, kw, pixel, coefficient):
    result = bytearray()
    for y in range(height):
        overlap_y = min(height, y + kh // 2 + 1) - max(0, y - kh // 2)
        for x in range(width):
            overlap_x = min(width, x + kw // 2 + 1) - max(0, x - kw // 2)
            value = normalize(pixel * coefficient * overlap_y * overlap_x)
            result.extend((value, value, value))
    return struct.pack("=II", height, width) + result


def run_case(program, directory, label, data, expected, timeout, reverse=False):
    input_path = directory / "input.bin"
    output_path = directory / "output.bin"
    input_path.write_bytes(data)
    output_path.unlink(missing_ok=True)
    arguments = ["-i", str(input_path), "-o", str(output_path)]
    if reverse:
        arguments = arguments[2:] + arguments[:2]
    started = time.perf_counter()
    try:
        completed = subprocess.run([str(program), *arguments], timeout=timeout,
                                   capture_output=True)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"exit code {completed.returncode}: "
                f"{completed.stderr.decode(errors='replace').strip()}")
        actual = output_path.read_bytes()
        if actual != expected:
            mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected))
                             if a != b), min(len(actual), len(expected)))
            raise RuntimeError(
                f"mismatch at byte {mismatch}; "
                f"actual length={len(actual)}, expected length={len(expected)}")
        return elapsed
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        failure = Path(__file__).resolve().parent / "test-failures" / label
        failure.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, failure / "input.bin")
        if output_path.exists():
            shutil.copyfile(output_path, failure / "actual.bin")
        (failure / "expected.bin").write_bytes(expected)
        raise RuntimeError(f"{label}: {error}; files saved in {failure}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    if args.cases < 0 or args.timeout <= 0:
        parser.error("cases must be nonnegative and timeout must be positive")
    program = args.program.resolve()
    if not program.is_file():
        parser.error(f"program does not exist: {program}")
    rng = random.Random(args.seed)
    print(f"Seed: {args.seed}; random cases: {args.cases}", flush=True)
    with tempfile.TemporaryDirectory(prefix="yadro-check-") as temporary:
        directory = Path(temporary)
        open_tests = Path(__file__).resolve().parent / "open-tests" / "open-tests"
        count = 0
        for input_path in sorted(open_tests.glob("*.in")):
            run_case(program, directory, "open-" + input_path.stem,
                     input_path.read_bytes(), input_path.with_suffix(".out").read_bytes(),
                     args.timeout)
            count += 1
        print(f"PASS: {count} open tests", flush=True)

        edges = [
            (1, 1, bytes((0, 127, 128)), 1, 1, [2]),
            (1, 1, bytes((0, 128, 255)), 1, 1, [-2]),
            (1, 1, bytes((0, 1, 255)), 1, 1, [1]),
            (1, 1, bytes((1, 241, 255)), 3, 3, [127] * 4 + [-1] + [127] * 4),
            (1, 7, rng.randbytes(21), 9, 3, [rng.randint(-128, 127) for _ in range(27)]),
            (7, 1, rng.randbytes(21), 3, 9, [rng.randint(-128, 127) for _ in range(27)]),
        ]
        # Maximum uint16 odd dimension, with only the central coefficient in bounds.
        for kh, kw in ((65535, 1), (1, 65535)):
            kernel = [-128] * (kh * kw)
            kernel[len(kernel) // 2] = 127
            edges.append((1, 1, bytes((0, 1, 255)), kh, kw, kernel))
        for index, case in enumerate(edges):
            expected = reference_python(*case)
            run_case(program, directory, f"edge-{index}", pack_input(*case),
                     expected, args.timeout, reverse=True)
        print(f"PASS: {len(edges)} edge cases", flush=True)

        numpy_module = None
        try:
            import numpy as numpy_module
        except ImportError:
            pass

        for index in range(args.cases):
            height, width = rng.randint(1, 12), rng.randint(1, 12)
            kh, kw = rng.randrange(1, 18, 2), rng.randrange(1, 18, 2)
            pixels = rng.randbytes(height * width * 3)
            kernel = [rng.randint(-128, 127) for _ in range(kh * kw)]
            if index % 10 == 0:
                kernel = [0] * (kh * kw)
            elif index % 10 == 1:
                kernel = [rng.choice((-128, -1, 1, 127))] * (kh * kw)
            if index % 11 == 0:
                pixels = bytes([rng.choice((0, 1, 241, 255))]) * len(pixels)
            case = height, width, pixels, kh, kw, kernel
            expected = reference_python(*case)
            if numpy_module is not None and index < 50:
                if reference_numpy(numpy_module, *case) != expected:
                    raise RuntimeError("Python and NumPy references disagree")
            run_case(program, directory, f"seed-{args.seed}-random-{index}",
                     pack_input(*case), expected, args.timeout, reverse=index % 2 == 0)
            if (index + 1) % 100 == 0:
                print(f"PASS: {index + 1}/{args.cases} random cases", flush=True)
        print(f"PASS: all {args.cases} random cases", flush=True)

        if not args.small_only:
            if numpy_module is None:
                print("SKIP: large random cases need NumPy (python -m pip install numpy)",
                      flush=True)
            else:
                for height, width, kh, kw in ((1024, 1536, 3, 5),
                                             (512, 768, 15, 15),
                                             (256, 320, 51, 31)):
                    label = f"large-{height}x{width}-kernel-{kh}x{kw}"
                    print(f"Running {label}...", flush=True)
                    case = (height, width, rng.randbytes(height * width * 3), kh, kw,
                            [rng.randint(-128, 127) for _ in range(kh * kw)])
                    expected = reference_numpy(numpy_module, *case)
                    elapsed = run_case(program, directory, label, pack_input(*case),
                                       expected, args.timeout)
                    print(f"PASS: {label}; program time {elapsed:.3f} s", flush=True)

            for coefficient in (127, -128):
                height = width = kh = kw = 259
                label = f"overflow-coefficient-{coefficient}"
                center_sum = 255 * coefficient * kh * kw
                print(f"Running {label}; center sum {center_sum}...", flush=True)
                case = (height, width, bytes([255]) * (height * width * 3),
                        kh, kw, [coefficient] * (kh * kw))
                expected = reference_constant(height, width, kh, kw, 255, coefficient)
                elapsed = run_case(program, directory, label, pack_input(*case),
                                   expected, args.timeout)
                print(f"PASS: {label}; program time {elapsed:.3f} s", flush=True)
    print("All enabled checks passed.", flush=True)


if __name__ == "__main__":
    main()
