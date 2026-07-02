use common::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct TopKSelection {
    pub token_index: usize,
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertAssignment {
    pub token_index: usize,
    pub topk_rank: usize,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertDispatch {
    pub expert_id: usize,
    pub assignments: Vec<ExpertAssignment>,
}

pub fn build_dispatch_plan(
    selections: &[TopKSelection],
    expert_count: usize,
) -> Result<Vec<ExpertDispatch>> {
    if expert_count == 0 {
        return Err(Error::moe("expert_count must be positive"));
    }

    let mut assignments_by_expert = vec![Vec::<ExpertAssignment>::new(); expert_count];
    for selection in selections {
        if selection.expert_ids.len() != selection.weights.len() {
            return Err(Error::moe(format!(
                "token {} expert_ids length {} does not match weights length {}",
                selection.token_index,
                selection.expert_ids.len(),
                selection.weights.len()
            )));
        }

        for (topk_rank, (&expert_id, &weight)) in selection
            .expert_ids
            .iter()
            .zip(selection.weights.iter())
            .enumerate()
        {
            if expert_id >= expert_count {
                return Err(Error::moe(format!(
                    "expert_id {expert_id} exceeds expert_count {expert_count}"
                )));
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(Error::moe(format!(
                    "invalid routing weight {weight} for token {} expert {expert_id}",
                    selection.token_index
                )));
            }

            assignments_by_expert[expert_id].push(ExpertAssignment {
                token_index: selection.token_index,
                topk_rank,
                weight,
            });
        }
    }

    Ok(assignments_by_expert
        .into_iter()
        .enumerate()
        .filter_map(|(expert_id, assignments)| {
            (!assignments.is_empty()).then_some(ExpertDispatch {
                expert_id,
                assignments,
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_plan_preserves_token_and_rank_indices() {
        let selections = vec![
            TopKSelection {
                token_index: 0,
                expert_ids: vec![1, 2],
                weights: vec![0.75, 0.25],
            },
            TopKSelection {
                token_index: 1,
                expert_ids: vec![2, 3],
                weights: vec![0.60, 0.40],
            },
        ];

        let plan = build_dispatch_plan(&selections, 4).unwrap();
        let expert_2 = plan
            .iter()
            .find(|dispatch| dispatch.expert_id == 2)
            .unwrap();

        assert_eq!(expert_2.assignments.len(), 2);
        assert_eq!(expert_2.assignments[0].token_index, 0);
        assert_eq!(expert_2.assignments[0].topk_rank, 1);
        assert_eq!(expert_2.assignments[1].token_index, 1);
        assert_eq!(expert_2.assignments[1].topk_rank, 0);
    }

    #[test]
    fn dispatch_plan_skips_unselected_experts() {
        let selections = vec![
            TopKSelection {
                token_index: 0,
                expert_ids: vec![0, 1],
                weights: vec![0.5, 0.5],
            },
            TopKSelection {
                token_index: 1,
                expert_ids: vec![0, 2],
                weights: vec![0.5, 0.5],
            },
            TopKSelection {
                token_index: 2,
                expert_ids: vec![0, 3],
                weights: vec![0.5, 0.5],
            },
        ];

        let plan = build_dispatch_plan(&selections, 4).unwrap();

        assert_eq!(plan.len(), 4);
        assert_eq!(plan[0].expert_id, 0);
        assert_eq!(plan[0].assignments.len(), 3);
        assert_eq!(plan[1].expert_id, 1);
        assert_eq!(plan[2].expert_id, 2);
        assert_eq!(plan[3].expert_id, 3);
    }
}
