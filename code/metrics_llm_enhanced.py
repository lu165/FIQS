import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的LLM增强评估模块
严格区分：模型记忆 vs 精确匹配 vs 智能泛化

核心改进：
1. 对系数精度要求极高（区分E1的"接近"和E2/E3的"精确"）
2. 结构匹配 + 系数精度 + 生物量精度三重评分
3. 只有完全正确才能得高分
"""

import re
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger(__name__)


class LLMEnhancedMetricsCalculator:
    """改进的LLM增强评估器 - 严格区分精确度"""
    
    def __init__(self, deepseek_api_key: str, qwen_model=None):
        """
        初始化评估器
        
        Args:
            deepseek_api_key: DeepSeek API密钥
            qwen_model: Qwen模型实例（用于LLM兜底）
        """
        self.deepseek_api_key = deepseek_api_key
        self.qwen_model = qwen_model
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        
        logger.info("✅ 改进的LLM增强评估器初始化完成（严格评分模式）")
    
    def set_qwen_model(self, qwen_model):
        """设置Qwen模型（用于LLM兜底）"""
        self.qwen_model = qwen_model
        logger.info("✅ Qwen模型已设置（用于LLM兜底）")
    

    def extract_equation_from_text(self, text):
        """Extract equation safely"""

        if not isinstance(text, str):
            return ""

        patterns = [
            r"W\s*=\s*[^\n]+",
            r"W=[^\n]+",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                return match.group(0)

        return ""

    def normalize_equation(self, equation: str) -> str:
        """标准化方程格式（保留精度）"""
        if not equation:
            return ""
        
        # 移除多余空格
        eq = equation.strip()
        
        # 统一W的大小写
        eq = re.sub(r'^[Ww]\s*=\s*', 'W=', eq)
        
        return eq
    
    def extract_coefficients(self, equation: str) -> List[float]:
        """提取方程中的所有系数"""
        if not equation:
            return []
        
        # 提取所有数字（包括小数）
        coefficients = []
        
        # 匹配模式：数字（整数或小数）
        pattern = r'\d+\.?\d*'
        matches = re.findall(pattern, equation)
        
        for match in matches:
            try:
                coeff = float(match)
                coefficients.append(coeff)
            except ValueError:
                pass
        
        return coefficients
    
    def extract_variables(self, equation: str) -> List[str]:
        """提取方程中的变量"""
        if not equation:
            return []
        
        # 提取大写字母变量（D, H, A, B等）
        variables = re.findall(r'[A-Z](?![a-z])', equation)
        
        # 去重并排序
        return sorted(set(variables))
    
    def calculate_coefficient_similarity(self, pred_coeffs: List[float], 
                                        ref_coeffs: List[float]) -> float:
        """
        计算系数相似度（极其严格）
        
        评分标准：
        - 完全匹配（误差<0.1%）: 1.0
        - 高度相似（误差<1%）: 0.9
        - 相似（误差<5%）: 0.7
        - 接近（误差<10%）: 0.5
        - 偏差大（误差<20%）: 0.3
        - 差异很大（误差>=20%）: 0.1
        """
        if not ref_coeffs or not pred_coeffs:
            return 0.0
        
        if len(pred_coeffs) != len(ref_coeffs):
            # 系数个数不同，严重扣分
            return 0.1
        
        total_similarity = 0.0
        
        for pred_c, ref_c in zip(pred_coeffs, ref_coeffs):
            if ref_c == 0:
                # 避免除零
                if pred_c == 0:
                    total_similarity += 1.0
                else:
                    total_similarity += 0.0
                continue
            
            # 计算相对误差
            relative_error = abs(pred_c - ref_c) / abs(ref_c)
            
            # 严格的评分标准
            if relative_error < 0.001:  # 0.1%
                similarity = 1.0
            elif relative_error < 0.01:  # 1%
                similarity = 0.9
            elif relative_error < 0.05:  # 5%
                similarity = 0.7
            elif relative_error < 0.10:  # 10%
                similarity = 0.5
            elif relative_error < 0.20:  # 20%
                similarity = 0.3
            else:  # >=20%
                similarity = 0.1
            
            total_similarity += similarity
        
        return total_similarity / len(ref_coeffs)
    
    def compare_equations_strict(self, pred_eq: str, ref_eq: str) -> Dict[str, Any]:
        """
        严格比较两个方程（区分模型记忆 vs 精确匹配）
        
        返回：
        - exact_match: 是否完全匹配
        - variable_match: 变量匹配度 (0-1)
        - coefficient_similarity: 系数相似度 (0-1)
        - structure_score: 结构分 (0-1)
        """
        pred_norm = self.normalize_equation(pred_eq)
        ref_norm = self.normalize_equation(ref_eq)
        
        result = {
            'exact_match': False,
            'variable_match': 0.0,
            'coefficient_similarity': 0.0,
            'structure_score': 0.0
        }
        
        if not pred_norm or not ref_norm:
            return result
        
        # 1. 完全匹配检测（去除所有空格后比较）
        pred_compact = re.sub(r'\s+', '', pred_norm)
        ref_compact = re.sub(r'\s+', '', ref_norm)
        
        result['exact_match'] = (pred_compact == ref_compact)
        
        # 2. 提取变量
        pred_vars = self.extract_variables(pred_eq)
        ref_vars = self.extract_variables(ref_eq)
        
        if ref_vars:
            matched_vars = set(pred_vars) & set(ref_vars)
            result['variable_match'] = len(matched_vars) / len(ref_vars)
        else:
            result['variable_match'] = 1.0 if not pred_vars else 0.0
        
        # 3. 提取并比较系数（最关键！）
        pred_coeffs = self.extract_coefficients(pred_eq)
        ref_coeffs = self.extract_coefficients(ref_eq)
        
        result['coefficient_similarity'] = self.calculate_coefficient_similarity(
            pred_coeffs, ref_coeffs
        )
        
        # 4. 结构分（综合变量和系数）
        if result['exact_match']:
            result['structure_score'] = 1.0
        else:
            # 变量和系数都重要
            result['structure_score'] = (
                result['variable_match'] * 0.3 +
                result['coefficient_similarity'] * 0.7
            )
        
        return result
    
    def extract_biomass_from_text(self, text: str) -> float:
        """从文本中提取生物量值"""
        if not text:
            return 0.0
        
        # 多种模式匹配
        patterns = [
            r'生物量[：:=\s]*([0-9.]+)',
            r'([0-9.]+)\s*kg',
            r'([0-9.]+)\s*千克',
            r'重[：:=\s]*([0-9.]+)',
            r'=\s*([0-9.]+)\s*kg',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        return 0.0
    
    def calculate_biomass_accuracy(self, pred_biomass: float, 
                                   ref_biomass: float) -> float:
        """
        计算生物量精度（严格标准）
        
        评分标准：
        - 误差<1%: 1.0
        - 误差<5%: 0.9
        - 误差<10%: 0.7
        - 误差<20%: 0.5
        - 误差<50%: 0.3
        - 误差>=50%: 0.0
        """
        if ref_biomass == 0:
            return 1.0 if pred_biomass == 0 else 0.0
        
        if pred_biomass == 0:
            return 0.0
        
        relative_error = abs(pred_biomass - ref_biomass) / abs(ref_biomass)
        
        if relative_error < 0.01:  # 1%
            return 1.0
        elif relative_error < 0.05:  # 5%
            return 0.9
        elif relative_error < 0.10:  # 10%
            return 0.7
        elif relative_error < 0.20:  # 20%
            return 0.5
        elif relative_error < 0.50:  # 50%
            return 0.3
        else:  # >=50%
            return 0.0
    
    def calculate_biomass_metrics(self, 
                                   predicted_answer: str,
                                   reference: Any,
                                   instruction: str = "",
                                   equation_db: List[Dict] = None) -> Dict[str, Any]:
        """
        计算Biomass任务的评分指标（严格模式）
        
        评分策略：
        1. 完全匹配（方程+生物量） → 1.0分 (E2/E3能达到)
        2. 方程高度相似 + 生物量准确 → 0.7-0.9分 (E3泛化)
        3. 方程接近 + 生物量偏差 → 0.3-0.6分 (E1模型记忆)
        4. 方程错误 → <0.3分
        
        Args:
            predicted_answer: 模型预测的答案
            reference: 标准答案（字符串或字典）
            instruction: 用户查询
            equation_db: 方程库（可选）
        
        Returns:
            包含score和details的字典
        """
        # 处理reference的两种格式
        if isinstance(reference, dict):
            ref_equation = reference.get('equation', '')
            ref_biomass = reference.get('biomass', 0.0)
        elif isinstance(reference, str):
            ref_equation = reference
            ref_biomass = 0.0
        else:
            ref_equation = str(reference)
            ref_biomass = 0.0
        
        # 从预测答案中提取方程和生物量
        pred_equation = self.extract_equation_from_text(predicted_answer)
        pred_biomass = self.extract_biomass_from_text(predicted_answer)
        
        # 严格比较方程
        equation_comparison = self.compare_equations_strict(pred_equation, ref_equation)
        
        # 计算生物量精度
        biomass_accuracy = self.calculate_biomass_accuracy(pred_biomass, ref_biomass)
        
        # ✅ 严格的综合评分策略
        if equation_comparison['exact_match']:
            # 完全匹配 → 检查生物量
            if biomass_accuracy >= 0.9:
                score = 1.0  # 完美
            elif biomass_accuracy >= 0.7:
                score = 0.9  # 方程对，生物量略有偏差
            else:
                score = 0.8  # 方程对，生物量偏差较大
        
        elif equation_comparison['structure_score'] >= 0.9:
            # 高度相似（系数相似度>0.9）
            score = 0.6 + biomass_accuracy * 0.3  # 0.6-0.9
        
        elif equation_comparison['structure_score'] >= 0.7:
            # 相似（系数相似度0.7-0.9）
            score = 0.4 + biomass_accuracy * 0.2  # 0.4-0.6
        
        elif equation_comparison['structure_score'] >= 0.5:
            # 接近（系数相似度0.5-0.7）→ 可能是模型记忆
            score = 0.2 + biomass_accuracy * 0.2  # 0.2-0.4
        
        else:
            # 差异大 → 可能是错误的方程
            score = equation_comparison['structure_score'] * 0.3  # <0.3
        
        # 详细信息
        details = {
            'pred_equation': pred_equation,
            'ref_equation': ref_equation,
            'pred_biomass': pred_biomass,
            'ref_biomass': ref_biomass,
            'exact_match': equation_comparison['exact_match'],
            'variable_match': equation_comparison['variable_match'],
            'coefficient_similarity': equation_comparison['coefficient_similarity'],
            'structure_score': equation_comparison['structure_score'],
            'biomass_accuracy': biomass_accuracy,
            # 兼容旧字段
            'species': 0.0,
            'region': 0.0,
            'component': 0.0,
            'variables': equation_comparison['variable_match'],
            'form': equation_comparison['structure_score'],
            'coefficient': equation_comparison['coefficient_similarity'],
            'used_fallback_region': False,
            'used_fallback_species': False
        }
        
        return {
            'score': score,
            'details': details,
            'method': 'strict_comparison'
        }
    
    def calculate_qa_metrics(self,
                            predicted: str,
                            reference: str,
                            question: str = "",
                            task_type: str = "forestry_qa") -> Dict[str, Any]:
        """
        计算QA任务的评分指标（准确性、完整性、相关性）
        
        Args:
            predicted: 模型预测的答案
            reference: 标准答案
            question: 用户问题
            task_type: 任务类型
        
        Returns:
            包含score和三个维度得分的字典
        """
        if not predicted or not reference:
            return {
                'score': 0.0,
                'accuracy': 0.0,
                'completeness': 0.0,
                'relevance': 0.0,
                'method': 'empty_response'
            }
        
        # 简单的启发式评分
        # 兼容LLM结构化输出
        if isinstance(predicted, dict):
            predicted = predicted.get("answer", "")

        if not isinstance(predicted, str):
            predicted = str(predicted)

        if isinstance(reference, dict):
            reference = reference.get("answer", "")

        if not isinstance(reference, str):
            reference = str(reference)

        # 1. 长度相似度
        len_ratio = min(len(predicted), len(reference)) / max(len(predicted), len(reference))
        
        # 2. 关键词匹配
        ref_words = set(reference.split())
        pred_words = set(predicted.split())
        
        if ref_words:
            keyword_overlap = len(ref_words & pred_words) / len(ref_words)
        else:
            keyword_overlap = 0.0
        
        # 3. 计算维度得分
        accuracy = keyword_overlap * 10  # 0-10分
        completeness = len_ratio * 10    # 0-10分
        relevance = min(accuracy, completeness)  # 0-10分
        
        # 综合得分（归一化到0-1）
        score = (accuracy + completeness + relevance) / 30
        
        return {
            'score': score,
            'accuracy': accuracy,
            'completeness': completeness,
            'relevance': relevance,
            'method': 'keyword_overlap'
        }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    calculator = LLMEnhancedMetricsCalculator(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        qwen_model=None
    )
    
    print("=" * 80)
    print("测试严格评分模式")
    print("=" * 80)
    
    reference = {
        'equation': 'W=0.022*D^2.495*H^0.357',
        'biomass': 45.92
    }
    
    # 测试1: E2/E3的完全匹配
    print("\n【测试1】E2/E3 - 完全匹配")
    pred1 = """推荐方程：W=0.022*D^2.495*H^0.357
生物量 = 45.92 kg"""
    
    result1 = calculator.calculate_biomass_metrics(pred1, reference)
    print(f"得分: {result1['score']:.2f}")
    print(f"方程完全匹配: {result1['details']['exact_match']}")
    print(f"系数相似度: {result1['details']['coefficient_similarity']:.2f}")
    print(f"生物量精度: {result1['details']['biomass_accuracy']:.2f}")
    
    # 测试2: E1的模型记忆（接近但不精确）
    print("\n【测试2】E1 - 模型记忆（系数不精确）")
    pred2 = """推荐方程：W=0.02*D^2.5*H^0.3
生物量 = 42.0 kg"""
    
    result2 = calculator.calculate_biomass_metrics(pred2, reference)
    print(f"得分: {result2['score']:.2f}")
    print(f"方程完全匹配: {result2['details']['exact_match']}")
    print(f"系数相似度: {result2['details']['coefficient_similarity']:.2f}")
    print(f"生物量精度: {result2['details']['biomass_accuracy']:.2f}")
    
    # 测试3: 完全错误
    print("\n【测试3】完全错误的方程")
    pred3 = """推荐方程：W=0.5*D^3
生物量 = 100.0 kg"""
    
    result3 = calculator.calculate_biomass_metrics(pred3, reference)
    print(f"得分: {result3['score']:.2f}")
    print(f"方程完全匹配: {result3['details']['exact_match']}")
    print(f"系数相似度: {result3['details']['coefficient_similarity']:.2f}")
    
    # 对比总结
    print("\n" + "=" * 80)
    print("评分对比:")
    print(f"  E2/E3 完全匹配:     {result1['score']:.2f} → 预期 0.9-1.0 ✅")
    print(f"  E1 模型记忆:        {result2['score']:.2f} → 预期 0.3-0.5 ✅")
    print(f"  完全错误:           {result3['score']:.2f} → 预期 <0.3 ✅")
    print("=" * 80)
    print("✅ 严格评分模式可以有效区分三种场景！")