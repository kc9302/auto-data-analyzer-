"""
Hybrid Multi-Signal Career Recommender & Component XAI Explainer.
Combines 4 distinct signals to calculate robust career recommendation scores:
1. ML Model Probability (XGBoost/LightGBM)
2. Alumni Pathway Tree Alignment Score (CareerPathwayTree)
3. Persona-Aware Feature Affinity Weight (PersonaFeatureWeightScout)
4. Market Popularity / Alumni Cohort Prevalence
5. Component-Level Explainable Attribution Breakdown (Why this job was recommended)
"""
import os
from typing import Dict, Any, List, Optional, Tuple


class HybridCareerRecommender:
    """
    Hybrid Multi-Signal Career Recommender.
    Harmonizes ML predictions, domain pathway trees, persona signals, and popularity
    while generating full explainability breakdown for end-user trust.
    """

    def __init__(
        self,
        weight_ml: float = 0.35,
        weight_tree: float = 0.30,
        weight_persona: float = 0.20,
        weight_popularity: float = 0.15
    ):
        # Normalize weights to sum to 1.0
        tot = weight_ml + weight_tree + weight_persona + weight_popularity or 1.0
        self.w_ml = weight_ml / tot
        self.w_tree = weight_tree / tot
        self.w_persona = weight_persona / tot
        self.w_pop = weight_popularity / tot

    def recommend_and_explain(
        self,
        student_id: Any,
        persona: str,
        ml_probabilities: Dict[str, float],
        tree_scores: Dict[str, float],
        persona_affinities: Dict[str, float],
        popularity_scores: Optional[Dict[str, float]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Calculates hybrid ensemble recommendation scores and produces component XAI breakdown.
        """
        popularity_scores = popularity_scores or {}

        # Collect all candidate job roles across all signals
        all_jobs = list(set(
            list(ml_probabilities.keys()) +
            list(tree_scores.keys()) +
            list(persona_affinities.keys()) +
            list(popularity_scores.keys())
        ))

        if not all_jobs:
            return {
                "student_id": student_id,
                "persona": persona,
                "recommendations": [],
                "summary": "추천 후보 직무가 없습니다."
            }

        recommendations = []
        for job in all_jobs:
            s_ml = float(ml_probabilities.get(job, 0.0))
            s_tree = float(tree_scores.get(job, 0.0))
            s_persona = float(persona_affinities.get(job, 0.0))
            s_pop = float(popularity_scores.get(job, 0.5))

            # Calculate weighted component contributions
            c_ml = self.w_ml * s_ml
            c_tree = self.w_tree * s_tree
            c_persona = self.w_persona * s_persona
            c_pop = self.w_pop * s_pop

            total_score = round(c_ml + c_tree + c_persona + c_pop, 4)

            # Attribution percentages (relative to total score)
            if total_score > 0:
                pct_ml = round((c_ml / total_score) * 100, 1)
                pct_tree = round((c_tree / total_score) * 100, 1)
                pct_persona = round((c_persona / total_score) * 100, 1)
                pct_pop = round((c_pop / total_score) * 100, 1)
            else:
                pct_ml = pct_tree = pct_persona = pct_pop = 25.0

            # Narrative explanation synthesis
            components = []
            if pct_tree >= 25.0:
                components.append(f"선배 합격자 이수 경로 일치율 {pct_tree}% (+{c_tree:.2f})")
            if pct_persona >= 20.0:
                components.append(f"'{persona}' 특화 역량 신호 {pct_persona}% (+{c_persona:.2f})")
            if pct_ml >= 25.0:
                components.append(f"AI 모델 예측 적합도 {pct_ml}% (+{c_ml:.2f})")
            if pct_pop >= 20.0:
                components.append(f"동문 선호 인기도 {pct_pop}% (+{c_pop:.2f})")

            explanation_narrative = " + ".join(components) if components else f"종합 역량 부합도 ({int(total_score*100)}점)"

            recommendations.append({
                "job_role": job,
                "hybrid_score": total_score,
                "fit_percentage": int(round(total_score * 100)),
                "raw_signals": {
                    "ml_prob": round(s_ml, 3),
                    "tree_match": round(s_tree, 3),
                    "persona_affinity": round(s_persona, 3),
                    "popularity": round(s_pop, 3)
                },
                "attribution_breakdown": {
                    "ml_contrib_pct": pct_ml,
                    "tree_contrib_pct": pct_tree,
                    "persona_contrib_pct": pct_persona,
                    "popularity_contrib_pct": pct_pop
                },
                "explanation_narrative": explanation_narrative
            })

        # Sort recommendations by hybrid_score descending
        recommendations = sorted(recommendations, key=lambda x: -x["hybrid_score"])[:top_k]

        top_job = recommendations[0]["job_role"] if recommendations else "미결정"
        summary_text = (
            f"학생({student_id}, {persona})에게 최적의 1순위 직무로 [{top_job}]를 추천합니다. "
            f"추천 사유: {recommendations[0]['explanation_narrative']}"
        )

        return {
            "student_id": student_id,
            "persona": persona,
            "top_job": top_job,
            "weights_used": {
                "ml": round(self.w_ml, 2),
                "tree": round(self.w_tree, 2),
                "persona": round(self.w_persona, 2),
                "popularity": round(self.w_pop, 2)
            },
            "recommendations": recommendations,
            "summary": summary_text
        }
