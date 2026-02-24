#![deny(warnings, unused, future_incompatible, nonstandard_style)]

pub mod ipa;
pub mod ipa_from_laurent;
pub mod helper;
pub mod r1cs;

use std::error::Error as ErrorTrait;
pub type Error = Box<dyn ErrorTrait>;