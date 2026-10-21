"""Tests for bit packing and unpacking."""

import torch
import pytest
from quantlab.bit_packer import BitPacker


def test_hand_calc_case3_exact_bytes():
    """Case 3: 4-bit [3, 12, 7, 1] gives bytes [195, 23]; 2-bit [2, 3, 0, 1] gives [78]."""
    packer = BitPacker()

    # 4-bit: 3 | (12 << 4) = 195; 7 | (1 << 4) = 23
    codes_4 = torch.tensor([3, 12, 7, 1], dtype=torch.uint8)
    packed_4 = packer.pack(codes_4, bits=4)
    assert packed_4.tolist() == [195, 23]

    unpacked_4 = packer.unpack(packed_4, bits=4, count=4)
    assert torch.equal(unpacked_4, codes_4)

    # 2-bit: 2 | (3<<2) | (0<<4) | (1<<6) = 2 + 12 + 0 + 64 = 78
    codes_2 = torch.tensor([2, 3, 0, 1], dtype=torch.uint8)
    packed_2 = packer.pack(codes_2, bits=2)
    assert packed_2.tolist() == [78]

    unpacked_2 = packer.unpack(packed_2, bits=2, count=4)
    assert torch.equal(unpacked_2, codes_2)


def test_bit_packer_round_trips_random():
    """Random round trips for 2, 4, 8 bits and for lengths that are not multiples."""
    packer = BitPacker()
    torch.manual_seed(42)

    for bits in (2, 4, 8):
        max_val = (1 << bits) - 1
        for count in (1, 3, 7, 15, 32, 33, 70, 128):
            codes = torch.randint(0, max_val + 1, (count,), dtype=torch.uint8)
            packed = packer.pack(codes, bits=bits)
            unpacked = packer.unpack(packed, bits=bits, count=count)
            assert torch.equal(codes, unpacked), f"Failed for bits={bits}, count={count}"


def test_bit_packer_invalid_bits():
    packer = BitPacker()
    with pytest.raises(ValueError):
        packer.pack(torch.zeros(4, dtype=torch.uint8), bits=3)
    with pytest.raises(ValueError):
        packer.unpack(torch.zeros(4, dtype=torch.uint8), bits=5, count=4)
