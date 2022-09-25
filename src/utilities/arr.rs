use std::{ops::{Sub, Add, Mul, Index, IndexMut, Deref}, iter::Sum};

use rand::{Rng, distributions::uniform::{SampleRange, SampleUniform}};
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arr<T, const SIZE: usize>(Vec<T>);

impl<T: Default, const SIZE: usize> Default for Arr<T, SIZE> {
    #[inline(always)]
    fn default() -> Self {
        let mut vec = Vec::new();
        vec.resize_with(SIZE, || T::default());
        Arr(vec)
    }
}

impl<T: Clone, const SIZE: usize> From<T> for Arr<T, SIZE> {
    #[inline(always)]
    fn from(v: T) -> Self {
        let mut vec = Vec::new();
        vec.resize_with(SIZE, || v.clone());
        Arr(vec)
    }
}

impl<T, const SIZE: usize> Arr<T, SIZE> {
    #[inline(always)]
    pub fn rand<R: Rng, U>(rng: &mut R, range: U) -> Self
    where 
    U: SampleRange<T> + Clone,
    T: SampleUniform {
        let mut result = Vec::with_capacity(SIZE);
        for _ in 0..SIZE {
            result.push(rng.gen_range(range.clone()));
        }
        Arr(result)
    }

    #[inline(always)]
    pub fn map<R, F>(&self, f: F) -> Arr<R, SIZE>
    where 
        F: FnMut(&T) -> R
    {
        Arr(self.0.iter().map(f).collect())
    }

    pub fn map_move<R, F>(self, f: F) -> Arr<R, SIZE>
    where 
        F: FnMut(T) -> R
    {
        Arr(self.0.into_iter().map(f).collect::<Vec<_>>())
    }


    #[inline(always)]
    pub fn map_copy<R, F>(&self, mut f: F) -> Arr<R, SIZE>
    where 
        F: FnMut(T) -> R,
        T: Clone
    {
        Arr(self.0.iter().map(|x| f(x.clone())).collect())
    }

    #[inline(always)]
    pub fn sum(self) -> T
    where T: Sum {
        self.0.into_iter().sum()
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.0.iter()
    }    

    #[inline(always)]
    pub unsafe fn uninitialized() -> Arr<T, SIZE> {
        let mut result = Vec::with_capacity(SIZE);
        result.set_len(SIZE);
        Arr(result)
    }
}

impl<T, const SIZE: usize> Deref for Arr<T, SIZE> {
    type Target = [T; SIZE];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.0.deref() as *const [T] as *const [T; SIZE]) }
    }
}

impl<T: Add<Output = T>, const SIZE: usize> Add for Arr<T, SIZE> {
    type Output = Arr<T, SIZE>;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut result = Vec::with_capacity(SIZE);
        for i in self.0.into_iter().zip(rhs.0) {
            result.push(i.0 + i.1)
        }
        Arr(result)
        //Arr(self.0.into_iter().zip(rhs.0).map(|(a, b)| a + b).collect())
    }
}

impl<T: Sub<Output = T>, const SIZE: usize> Sub for Arr<T, SIZE> {
    type Output = Arr<T, SIZE>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = Vec::with_capacity(SIZE);
        for i in self.0.into_iter().zip(rhs.0) {
            result.push(i.0 - i.1)
        }
        Arr(result)
        //Arr(self.0.into_iter().zip(rhs.0).map(|(a, b)| a - b).collect())
    }
}

impl<T: Mul<Output = T>, const SIZE: usize> Mul for Arr<T, SIZE> {
    type Output = Arr<T, SIZE>;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = Vec::with_capacity(SIZE);
        for i in self.0.into_iter().zip(rhs.0) {
            result.push(i.0 * i.1)
        }
        Arr(result)
        //Arr(self.0.into_iter().zip(rhs.0.into_iter()).map(|(a, b)| a * b).collect())
    }
}

impl<T: Default, const SIZE: usize> Sum for Arr<T, SIZE> 
where Self: Add<Output = Self> {
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result = Default::default();
        for i in iter {
            result = result + i;
        }
        result
        //iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

impl<T, const SIZE: usize> Index<usize> for Arr<T, SIZE> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const SIZE: usize> IndexMut<usize> for Arr<T, SIZE> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}


impl<T, const SIZE: usize> TryFrom<Vec<T>> for Arr<T, SIZE> {
    type Error = Vec<T>;

    #[inline(always)]
    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        if value.len() == SIZE {
            Ok(Arr(value))
        } else {
            Err(value)
        }
    }
}

impl<T: Clone, const SIZE: usize> From<&[T; SIZE]> for Arr<T, SIZE> {
    #[inline(always)]
    fn from(value: &[T; SIZE]) -> Self {
        Self(value[..].into())
    }
}

impl<T, const SIZE: usize> From<[T; SIZE]> for Arr<T, SIZE> {
    #[inline(always)]
    fn from(value: [T; SIZE]) -> Self {
        Self(value.into())
    }
}

impl<T, const SIZE: usize> IntoIterator for Arr<T, SIZE> {
    type Item = T;

    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}