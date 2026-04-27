use core::mem::MaybeUninit;
use core::slice;

use crate::Hasher as _;

/// 32-bit `MurmurHash3` hasher
///
/// # Examples
///
/// ```
/// use core::hash::Hasher as _;
/// use hash32::{Hasher as _, Murmur3Hasher};
///
/// let mut hasher: Murmur3Hasher = Default::default();
/// hasher.write(b"Hello, World!");
///
/// println!("Hash is {:x}!", hasher.finish32());
/// ```
#[derive(Debug, Clone)]
pub struct Murmur3Hasher {
    buf: Buffer,
    index: Index,
    processed: u32,
    state: State,
}

#[derive(Debug, Clone)]
struct State(u32);

#[derive(Debug, Clone, Copy)]
#[repr(align(4))]
struct Buffer {
    bytes: [MaybeUninit<u8>; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Index {
    _0,
    _1,
    _2,
    _3,
}

impl Index {
    fn usize(self) -> usize {
        match self {
            Self::_0 => 0,
            Self::_1 => 1,
            Self::_2 => 2,
            Self::_3 => 3,
        }
    }
}

impl From<usize> for Index {
    fn from(x: usize) -> Self {
        match x % 4 {
            0 => Self::_0,
            1 => Self::_1,
            2 => Self::_2,
            3 => Self::_3,
            _ => unreachable!(),
        }
    }
}

impl Murmur3Hasher {
    /// # Safety
    ///
    /// The caller must ensure that `self.index.usize() + buf.len() <= 4`.
    unsafe fn push(&mut self, buf: &[u8]) {
        let start = self.index.usize();
        let len = buf.len();
        // NOTE(unsafe) avoid calling `memcpy` on a 0-3 byte copy
        // self.buf.bytes[start..start+len].copy_from(buf);
        for i in 0..len {
            // SAFETY:
            // 1. `start + x` is less than or equal to `start + len`, which is `<=` 4 by the
            //    function precondition, so `self.buf.bytes.get_unchecked_mut(start + i)` is in bounds.
            // 2. `i` is within the range `0..len`, which matches the length of `buf`, so
            //    `buf.get_unchecked(i)` is in bounds.
            unsafe {
                self.buf
                    .bytes
                    .get_unchecked_mut(start + i)
                    .write(*buf.get_unchecked(i));
            }
        }
        self.index = Index::from(start + len);
    }
}

impl Default for Murmur3Hasher {
    fn default() -> Self {
        Self {
            buf: Buffer {
                bytes: [MaybeUninit::uninit(); 4],
            },
            index: Index::_0,
            processed: 0,
            state: State(0),
        }
    }
}

impl crate::Hasher for Murmur3Hasher {
    fn finish32(&self) -> u32 {
        // tail
        let mut state = match self.index {
            Index::_3 => {
                let mut block = 0;
                // SAFETY: `self.index == 3` indicates that exactly 3 bytes
                // have been written and initialized via previous `push()` or `write()` calls.
                unsafe {
                    block ^= u32::from(self.buf.bytes[2].assume_init()) << 16;
                    block ^= u32::from(self.buf.bytes[1].assume_init()) << 8;
                    block ^= u32::from(self.buf.bytes[0].assume_init());
                }
                self.state.0 ^ pre_mix(block)
            }
            Index::_2 => {
                let mut block = 0;
                // SAFETY: `self.index == 2` indicates that exactly 2 bytes
                // have been written and initialized via previous `push()` or `write()` calls.
                unsafe {
                    block ^= u32::from(self.buf.bytes[1].assume_init()) << 8;
                    block ^= u32::from(self.buf.bytes[0].assume_init());
                }
                self.state.0 ^ pre_mix(block)
            }
            Index::_1 => {
                let mut block = 0;
                // SAFETY: `self.index == 1` indicates that exactly 1 byte
                // has been written and initialized via previous `push()` or `write()` calls.
                unsafe {
                    block ^= u32::from(self.buf.bytes[0].assume_init());
                }
                self.state.0 ^ pre_mix(block)
            }
            Index::_0 => self.state.0,
        };

        // finalization mix
        state ^= self.processed;
        state ^= state >> 16;
        state = state.wrapping_mul(0x85ebca6b);
        state ^= state >> 13;
        state = state.wrapping_mul(0xc2b2ae35);
        state ^= state >> 16;

        state
    }
}

impl core::hash::Hasher for Murmur3Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        self.processed += len as u32;

        let body = if self.index == Index::_0 {
            // CASE 1
            bytes
        } else {
            let index = self.index.usize();
            if len + index >= 4 {
                // CASE 2

                // we can complete a block using the data left in the buffer
                // NOTE(unsafe) avoid panicking branch (`slice_index_len_fail`)
                // let (head, body) = bytes.split_at(4 - index);
                let mid = 4 - index;
                // SAFETY: By condition `len + index >= 4`, we have `len >= 4 - index` (= `mid`).
                // Hence `bytes` contains at least `mid` valid bytes to construct a slice from its pointer.
                let head = unsafe { slice::from_raw_parts(bytes.as_ptr(), mid) };
                // SAFETY: `bytes.as_ptr().add(mid)` stays within bounds of the original `bytes` slice
                // because `mid <= len`. The remaining length is `len - mid`.
                let body = unsafe { slice::from_raw_parts(bytes.as_ptr().add(mid), len - mid) };

                // NOTE(unsafe) avoid calling `memcpy` on a 0-3 byte copy
                // self.buf.bytes[index..].copy_from_slice(head);
                for i in 0..4 - index {
                    // SAFETY:
                    // 1. `index + i < index + (4 - index) = 4`, so it's in bounds of `self.buf.bytes`.
                    // 2. `i < 4 - index = mid`, so it's in bounds of `head`.
                    unsafe {
                        self.buf
                            .bytes
                            .get_unchecked_mut(index + i)
                            .write(*head.get_unchecked(i));
                    }
                }

                self.index = Index::_0;

                // SAFETY: the loop above just wrote bytes [index..4], and prior push() calls
                // wrote bytes [0..index], so all 4 bytes are initialized.
                // The transmute from `&[MaybeUninit<u8>; 4]` to `&[u8; 4]` is valid.
                let block: &[u8; 4] = unsafe { core::mem::transmute(&self.buf.bytes) };
                self.state.process_block(block);

                body
            } else {
                // CASE 3
                bytes
            }
        };

        for block in body.chunks(4) {
            if block.len() == 4 {
                // SAFETY: By condition `block.len() == 4`, direct cast to `&[u8; 4]` is valid,
                // as the slice pointer is valid for 4 contiguous readable bytes,
                // and arrays of `u8` require only 1-byte alignment.
                self.state
                    .process_block(unsafe { &*(block.as_ptr().cast::<[u8; 4]>()) });
            } else {
                // SAFETY:
                // 1. In CASE 1 and CASE 2 above, `self.index.usize() == 0`, so `self.index.usize() + block.len() < 4`.
                // 2. In CASE 3, the condition for this branch ensures that `self.index.usize() + bytes.len() < 4`,
                //    and since `block == body == bytes`, `self.index.usize() + block.len() < 4`.
                // In all cases, the precondition for `self.push()` is upheld.
                unsafe {
                    self.push(block);
                }
            }
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.finish32().into()
    }
}

const C1: u32 = 0xcc9e2d51;
const C2: u32 = 0x1b873593;
const R1: u32 = 15;

impl State {
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn process_block(&mut self, block: &[u8; 4]) {
        self.0 ^= pre_mix(u32::from_le_bytes(*block));
        self.0 = self.0.rotate_left(13);
        self.0 = 5u32.wrapping_mul(self.0).wrapping_add(0xe6546b64);
    }
}

fn pre_mix(mut block: u32) -> u32 {
    block = block.wrapping_mul(C1);
    block = block.rotate_left(R1);
    block = block.wrapping_mul(C2);
    block
}
