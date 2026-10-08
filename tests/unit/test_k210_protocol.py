"""K210 封包解析：任意切斷、雜訊、錯誤長度／JSON／JPEG、序號跳號、文字交錯。"""

from __future__ import annotations

import random

import pytest

from edgelens.sources.k210.protocol import MAGIC, Packet, PacketParser, TextChunk, encode_packet

JPEG = b"\xff\xd8" + bytes(range(50)) + b"\xff\xd9"


def run(chunks, parser=None):
    p = parser or PacketParser()
    out = []
    for c in chunks:
        out += p.feed(c)
    return p, out


def packets(items):
    return [i for i in items if isinstance(i, Packet)]


def text_of(items):
    return "".join(i.text for i in items if isinstance(i, TextChunk))


def stream(n=5, start=0):
    return b"".join(encode_packet({"seq": start + i, "w": 320, "h": 240}, JPEG) for i in range(n))


def test_basic_packets():
    p, items = run([stream(3)])
    assert [x.meta["seq"] for x in packets(items)] == [0, 1, 2]
    assert all(x.jpeg == JPEG for x in packets(items))
    assert p.stats == {"packets": 3, "bad_frames": 0, "junk_bytes": 0, "lost_seq": 0}


def test_split_at_every_byte_position():
    data = stream(3)
    for cut in range(1, len(data)):
        p, items = run([data[:cut], data[cut:]])
        assert [x.meta["seq"] for x in packets(items)] == [0, 1, 2], cut
        assert p.junk_bytes == 0 and p.bad_frames == 0 and p.lost_seq == 0, cut


def test_one_byte_at_a_time():
    data = b"hello\r\n" + stream(2) + b"tail text\r\n"
    p, items = run([bytes([b]) for b in data])
    assert len(packets(items)) == 2
    assert text_of(items) == "hello\r\ntail text\r\n"
    assert p.junk_bytes == 0


def test_data_only_packet():
    _, items = run([encode_packet({"seq": 0, "objs": []}, b"")])
    assert packets(items)[0].jpeg == b"" and packets(items)[0].meta["objs"] == []


def test_text_interleaved_with_packets_keeps_order():
    data = b"CAPS {}\r\n" + stream(1) + b"ERROR at seq 1\r\n" + stream(1, start=1)
    _, items = run([data])
    kinds = [type(i).__name__ for i in items]
    assert kinds == ["TextChunk", "Packet", "TextChunk", "Packet"]
    assert items[0].text == "CAPS {}\r\n" and items[2].text == "ERROR at seq 1\r\n"


def test_utf8_text_split_inside_character():
    data = "錯誤：找不到模型\r\n".encode() + stream(1)
    for cut in range(1, len(data)):
        p, items = run([data[:cut], data[cut:]])
        assert text_of(items) == "錯誤：找不到模型\r\n", cut
        assert p.junk_bytes == 0 and len(packets(items)) == 1


def test_oversized_lengths_resync():
    import struct

    from edgelens.sources.k210.protocol import HEADER

    bad_meta = MAGIC + HEADER.pack(5000, 10) + b"x" * 20
    bad_jpeg = MAGIC + HEADER.pack(10, 300000) + b"y" * 20
    p, items = run([bad_meta + bad_jpeg + stream(2)])
    assert [x.meta["seq"] for x in packets(items)] == [0, 1]
    assert p.junk_bytes >= 8
    assert struct.calcsize("<HI") == HEADER.size


def test_bad_json_resync():
    from edgelens.sources.k210.protocol import HEADER

    body = b"{not json!!"
    broken = MAGIC + HEADER.pack(len(body), 0) + body
    p, items = run([broken + stream(2)])
    assert [x.meta["seq"] for x in packets(items)] == [0, 1]
    assert p.junk_bytes >= 4


def test_json_not_object_is_rejected():
    from edgelens.sources.k210.protocol import HEADER

    body = b"[1,2,3]"
    p, items = run([MAGIC + HEADER.pack(len(body), 0) + body + stream(1)])
    assert len(packets(items)) == 1


@pytest.mark.parametrize("bad", [b"\xff\xd8" + b"abc", b"abc\xff\xd9", b"not a jpeg", b"\xff\xd9\xff\xd8"])
def test_bad_jpeg_counts_bad_frame(bad):
    data = encode_packet({"seq": 0}, bad) + encode_packet({"seq": 1}, JPEG)
    p, items = run([data])
    assert [x.meta["seq"] for x in packets(items)] == [1]
    assert p.bad_frames == 1 and p.packets == 1


def test_seq_gap_counts_lost():
    data = encode_packet({"seq": 0}, JPEG) + encode_packet({"seq": 4}, JPEG) + encode_packet({"seq": 5}, JPEG)
    p, _ = run([data])
    assert p.lost_seq == 3


def test_bad_jpeg_still_tracks_seq():
    data = encode_packet({"seq": 0}, JPEG) + encode_packet({"seq": 1}, b"junk") + encode_packet({"seq": 2}, JPEG)
    p, _ = run([data])
    assert p.lost_seq == 0 and p.bad_frames == 1


def test_seq_restart_does_not_count_lost():
    data = encode_packet({"seq": 100}, JPEG) + encode_packet({"seq": 0}, JPEG) + encode_packet({"seq": 1}, JPEG)
    p, _ = run([data])
    assert p.lost_seq == 0


def test_binary_noise_counts_as_junk_not_text():
    rng = random.Random(1)
    noise = bytes(rng.randrange(0x80, 0x100) | 0x80 for _ in range(200)) + b"\x01\x02\x00"
    # 0x80 以上的隨機位元組幾乎不可能是合法 UTF-8；就算是，也必須不影響封包
    p, items = run([noise + stream(2)])
    assert len(packets(items)) == 2
    assert p.junk_bytes + len(text_of(items).encode()) >= len(noise) - 3


def test_random_noise_between_packets_recovers():
    rng = random.Random(7)
    for trial in range(30):
        parts, expect = [], []
        for i in range(8):
            parts.append(bytes(rng.randrange(256) for _ in range(rng.randrange(0, 60))).replace(MAGIC[:2], b"zz"))
            parts.append(encode_packet({"seq": i}, JPEG))
            expect.append(i)
        data = b"".join(parts)
        cuts = sorted(rng.sample(range(1, len(data)), 5))
        chunks = [data[a:b] for a, b in zip([0, *cuts], [*cuts, len(data)], strict=True)]
        _, items = run(chunks)
        assert [x.meta["seq"] for x in packets(items)] == expect, trial


def test_truncated_stream_waits_for_more():
    data = stream(1)
    p, items = run([data[:-5]])
    assert packets(items) == []
    items = p.feed(data[-5:])
    assert len(packets(items)) == 1


def test_reset_clears_buffer_but_keeps_counters():
    p, _ = run([encode_packet({"seq": 0}, b"bad"), stream(1)[:10]])
    assert p.bad_frames == 1
    p.reset()
    _, items = run([stream(1)], p)
    assert len(packets(items)) == 1 and p.bad_frames == 1
    p.reset_counters()
    assert p.stats == {"packets": 0, "bad_frames": 0, "junk_bytes": 0, "lost_seq": 0}


def test_eot_bytes_removed_from_text():
    _, items = run([b"\x04>>> ok\r\n"])
    assert text_of(items) == ">>> ok\r\n"


def test_encode_packet_layout():
    raw = encode_packet({"a": 1}, b"\xff\xd8\xff\xd9")
    assert raw.startswith(MAGIC) and raw.endswith(b"\xff\xd9")
