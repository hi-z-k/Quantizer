"""Bit packer: packs and unpacks integer codes into compact byte arrays.

Sequential layout where the first value sits in the lowest bits of the
first byte. Supports 2, 4, and 8 bits per code.

Example (4-bit, codes [3, 12, 7, 1]):
    Byte 0: code[0]=3 in bits 0-3, code[1]=12 in bits 4-7 → 3 | (12 << 4) = 3 + 192 = 195
    Byte 1: code[2]=7 in bits 0-3, code[3]=1  in bits 4-7 → 7 | (1 << 4)  = 7 + 16  = 23
    Result: bytes [195, 23]

Example (2-bit, codes [2, 3, 0, 1]):
    Byte 0: 2 | (3<<2) | (0<<4) | (1<<6) = 2 + 12 + 0 + 64 = 78
    Result: bytes [78]

If the count is not a multiple of (8 // bits), pad with zeros on pack
and cut to the stored count on unpack.
"""

from __future__ import annotations

import torch

from quantlab.interfaces import Packer


class BitPacker(Packer):
    """Packs unsigned integer codes into uint8 bytes.

    The first code occupies the lowest bits of the first byte.
    Supports 2-bit, 4-bit, and 8-bit packing.
    """

    def pack(self, codes: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack 1-D unsigned integer codes into uint8 bytes.

        Args:
            codes: 1-D tensor of unsigned integer codes (values fit in ``bits``).
            bits: Bits per code (2, 4, or 8).

        Returns:
            1-D uint8 tensor.

        Raises:
            ValueError: If bits is not 2, 4, or 8.
        """
        if bits not in (2, 4, 8):
            raise ValueError(f"BitPacker supports 2, 4, 8 bits, got {bits}")

        codes = codes.to(torch.uint8)

        if bits == 8:
            return codes

        codes_per_byte = 8 // bits
        n = codes.numel()
        # Pad with zeros if not a multiple
        remainder = n % codes_per_byte
        if remainder != 0:
            pad_count = codes_per_byte - remainder
            codes = torch.cat([codes, torch.zeros(pad_count, dtype=torch.uint8)])

        n_bytes = codes.numel() // codes_per_byte
        codes = codes.reshape(n_bytes, codes_per_byte)

        packed = torch.zeros(n_bytes, dtype=torch.uint8)
        for i in range(codes_per_byte):
            packed |= codes[:, i] << (i * bits)
        return packed

    def unpack(self, packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
        """Unpack codes from packed uint8 bytes.

        Args:
            packed: 1-D uint8 tensor from ``pack()``.
            bits: Bits per code (2, 4, or 8).
            count: Original number of codes (before any padding).

        Returns:
            1-D uint8 tensor of ``count`` codes.

        Raises:
            ValueError: If bits is not 2, 4, or 8.
        """
        if bits not in (2, 4, 8):
            raise ValueError(f"BitPacker supports 2, 4, 8 bits, got {bits}")

        if bits == 8:
            return packed[:count]

        codes_per_byte = 8 // bits
        mask = (1 << bits) - 1

        # Expand each byte into its codes
        all_codes = []
        for i in range(codes_per_byte):
            all_codes.append((packed >> (i * bits)) & mask)
        # Interleave: byte0_code0, byte0_code1, ..., byte1_code0, ...
        stacked = torch.stack(all_codes, dim=1)  # [n_bytes, codes_per_byte]
        flat = stacked.reshape(-1)
        return flat[:count]
