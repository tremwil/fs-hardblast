use std::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Range},
};

#[derive(Debug)]
pub struct ConstVec<T, const N: usize> {
    buffer: [MaybeUninit<T>; N],
    len: usize,
}

impl<T: Clone, const N: usize> Clone for ConstVec<T, N> {
    fn clone(&self) -> Self {
        let mut clone = Self::new();
        for item in self.iter() {
            clone.push(item.clone());
        }
        clone
    }
}

impl<T: Copy, const N: usize> Copy for ConstVec<T, N> {}

impl<T, const N: usize> ConstVec<T, N> {
    pub const fn new() -> Self {
        Self {
            buffer: [const { MaybeUninit::uninit() }; N],
            len: 0,
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn push(&mut self, value: T) {
        assert!(self.len < N, "ConstVec full");

        self.buffer[self.len].write(value);
        self.len += 1;
    }

    pub const fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr().cast(), self.len) }
    }

    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr().cast(), self.len) }
    }

    pub const fn index(&self, index: usize) -> &T {
        &self.as_slice()[index]
    }

    pub const fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.as_mut_slice()[index]
    }
}

impl<T: Copy, const N: usize> ConstVec<T, N> {
    pub const fn from_slice_range(slice: &[T], range: Range<usize>) -> Self {
        assert!(
            range.start < slice.len() && range.end <= slice.len(),
            "range out of bounds"
        );
        let len = range.end - range.start;
        assert!(len <= N, "range too big");

        let mut this = Self::new();
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice.as_ptr().add(range.start),
                this.buffer.as_mut_ptr().cast(),
                len,
            );
        }

        this.len = len;
        this
    }
}

impl<T, const N: usize> Deref for ConstVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for ConstVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}
