
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use merlin::Transcript;

// An implementation of Fiat-Shamir from merlin
pub trait ProofTranscript<P: Pairing> {
  fn append_protocol_name(&mut self, protocol_name: &'static [u8]);
  fn append_scalar(&mut self, label: &'static [u8], scalar: &P::ScalarField);
  fn append_scalars(&mut self, label: &'static [u8], scalars: &[P::ScalarField]);
  fn append_point(&mut self, label: &'static [u8], point: &P::G1);
  fn append_points(&mut self, label: &'static [u8], points: &[P::G1]);
  fn challenge_scalar(&mut self, label: &'static [u8]) -> P::ScalarField;
  fn challenge_vector(&mut self, label: &'static [u8], len: usize) -> Vec<P::ScalarField>;
}

impl<P: Pairing> ProofTranscript<P> for Transcript {
  fn append_protocol_name(&mut self, protocol_name: &'static [u8]) {
    self.append_message(b"protocol-name", protocol_name);
  }

  fn append_scalar(&mut self, label: &'static [u8], scalar: &P::ScalarField) {
    let mut buf = vec![];
    scalar.serialize_uncompressed(&mut buf).unwrap();
    self.append_message(label, &buf);
  }

  fn append_scalars(&mut self, label: &'static [u8], scalars: &[P::ScalarField]) {
    self.append_message(label, b"begin_append_vector");
    for item in scalars.iter() {
      <Self as ProofTranscript<P>>::append_scalar(self, label, item);
    }
    self.append_message(label, b"end_append_vector");
  }

  fn append_point(&mut self, label: &'static [u8], point: &P::G1) {
    let mut buf = vec![];
    point.serialize_uncompressed(&mut buf).unwrap();
    self.append_message(label, &buf);
  }

  fn append_points(&mut self, label: &'static [u8], points: &[P::G1]) {
    self.append_message(label, b"begin_append_vector");
    for item in points.iter() {
      <Self as ProofTranscript<P>>::append_point(self, label, item);
    }
    self.append_message(label, b"end_append_vector");
  }

  fn challenge_scalar(&mut self, label: &'static [u8]) -> P::ScalarField {
    let mut buf = [0u8; 64];
    self.challenge_bytes(label, &mut buf);
    P::ScalarField::from_le_bytes_mod_order(&buf)
  }

  fn challenge_vector(&mut self, label: &'static [u8], len: usize) -> Vec<P::ScalarField> {
    (0..len)
      .map(|_i| <Self as ProofTranscript<P>>::challenge_scalar(self, label))
      .collect::<Vec<P::ScalarField>>()
  }
}