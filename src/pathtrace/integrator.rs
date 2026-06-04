//! Integrator selection — MIS+NEE vs. pure BSDF.
//!
//! M2 set up runtime-selectable samplers. M3 needs a second axis to
//! tell the convergence story: at equal samples-per-pixel, MIS + NEE
//! should beat a pure BSDF integrator on RMSE-vs-reference. Both
//! integrators live in the same WGSL shader, dispatched from a single
//! `integrator_kind` uniform.
//!
//! - **MIS + NEE.** Next-event estimation at every bounce, combined with
//!   BSDF sampling via the power heuristic. The low-variance default
//!   that M1 shipped.
//! - **Pure BSDF.** No next-event estimation; light is only ever found
//!   when a BSDF-sampled ray happens to hit the emitter. Every emission
//!   contribution is added unweighted. The variance baseline.

use std::str::FromStr;

/// Which integrator the path tracer uses.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IntegratorKind {
    /// Multiple-importance-sampled next-event estimation (the M1 default).
    #[default]
    MisNee = 0,
    /// Pure BSDF path tracing — variance baseline for the M3 study.
    Bsdf = 1,
}

impl IntegratorKind {
    /// Discriminant exactly as written to the WGSL uniform.
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Short label used in CSV output and CLI args.
    pub fn label(self) -> &'static str {
        match self {
            IntegratorKind::MisNee => "misnee",
            IntegratorKind::Bsdf => "bsdf",
        }
    }
}

impl FromStr for IntegratorKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "misnee" | "mis+nee" | "mis_nee" | "mis" => Ok(IntegratorKind::MisNee),
            "bsdf" => Ok(IntegratorKind::Bsdf),
            other => Err(format!(
                "unknown integrator: {other} (expected misnee|bsdf)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_match_wgsl_constants() {
        assert_eq!(IntegratorKind::MisNee.as_u32(), 0);
        assert_eq!(IntegratorKind::Bsdf.as_u32(), 1);
    }

    #[test]
    fn from_str_accepts_synonyms() {
        assert_eq!(
            "misnee".parse::<IntegratorKind>().unwrap(),
            IntegratorKind::MisNee
        );
        assert_eq!(
            "MIS+NEE".parse::<IntegratorKind>().unwrap(),
            IntegratorKind::MisNee
        );
        assert_eq!(
            "bsdf".parse::<IntegratorKind>().unwrap(),
            IntegratorKind::Bsdf
        );
        assert!("xxx".parse::<IntegratorKind>().is_err());
    }

    #[test]
    fn labels_are_csv_friendly() {
        assert_eq!(IntegratorKind::MisNee.label(), "misnee");
        assert_eq!(IntegratorKind::Bsdf.label(), "bsdf");
    }
}
