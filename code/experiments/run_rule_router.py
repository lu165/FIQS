import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版：主实验E2 - 固定规则路由（业界标准方法）
使用严格字符串匹配的BiomassEstimatorBasic
功能：基于关键词匹配的规则路由系统
日期：2026-05-25
"""

import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

# 添加项目路径
sys.path.insert(0, '.')

# 导入LLM增强评估模块
from metrics_llm_enhanced import LLMEnhancedMetricsCalculator

# 导入测试数据加载器
from test_data_loader import TestDataLoader

# 配置日志
log_dir = "./experiments/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'E2_fixed_rule_routing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BiomassKeywordMatcher:
    """生物量任务关键词匹配器"""
    
    BIOMASS_KEYWORDS = [
        '生物量', '碳储量', '蓄积量', '碳汇', '碳存储',
        '地上生物量', '地下生物量', '总生物量',
        'DBH', 'dbh', 'D=', 'd=',
        '胸径', '树高', 'H=', 'h=',
        '直径', '高度',
        '方程', '计算', '估算', '测算',
        '模型', '公式',
        '吨', 'kg', 't/hm', 't/ha',
        'kg/株', '千克',
        '思茅松', '马尾松', '杉木', '云南松',
        '干', '枝', '叶', '根',
    ]
    
    def __init__(self):
        self.keywords_lower = [kw.lower() for kw in self.BIOMASS_KEYWORDS]
        logger.info(f"✅ 生物量关键词库初始化完成，共{len(self.BIOMASS_KEYWORDS)}个关键词")
    
    def is_biomass_query(self, text: str) -> bool:
        """判断是否为生物量查询"""
        text_lower = text.lower()
        
        for keyword in self.keywords_lower:
            if keyword in text_lower:
                return True
        
        return False
    
    def get_matched_keywords(self, text: str) -> List[str]:
        """获取匹配到的关键词"""
        text_lower = text.lower()
        matched = []
        
        for keyword in self.keywords_lower:
            if keyword in text_lower:
                matched.append(keyword)
        
        return matched


class FixedRuleRoutingExperiment:
    """E2固定规则路由实验主类"""
    
    def __init__(self):
        # 路径配置
        self.base_model_path = "./models/Qwen1.5-7B-Chat"
        self.lora_path = "./checkpoints/Qwen1.5-7B-Chat_lora/checkpoint-74170"
        
        # ✅ 修改：使用新的测试数据路径
        self.biomass_test_file = "./data/最新生物量测试集构建_v7_with_neg.json"
        self.equation_db_path = "./data/林业方程_标准化.json"
        
        # DeepSeek API配置
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 输出配置
        self.output_dir = "./experiments/results/main"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.output_file = os.path.join(self.output_dir, "E2_fixed_rule_routing_results.json")
        self.detailed_file = os.path.join(self.output_dir, "E2_fixed_rule_routing_detailed.jsonl")
        self.routing_log_file = os.path.join(self.output_dir, "E2_routing_log.jsonl")
        
        # 组件
        self.keyword_matcher = BiomassKeywordMatcher()
        self.qwen_model = None
        self.general_chat = None
        self.forestry_qa = None
        self.biomass_estimator = None
        
        # 评估器
        self.metrics_calculator = LLMEnhancedMetricsCalculator(
            deepseek_api_key=self.deepseek_api_key,
            qwen_model=None
        )
        
        # 统计数据结构
        self.results = []
        self.task_stats = defaultdict(lambda: {
            'total': 0,
            'route_correct': 0,
            'answer_correct': 0,
            'scores': [],
            'response_lengths': [],
            'processing_times': [],
            'routing_times': [],
            'generation_times': [],
            'species_scores': [],
            'region_scores': [],
            'component_scores': [],
            'variable_scores': [],
            'form_scores': [],
            'coefficient_scores': [],
            'fallback_region_count': 0,
            'fallback_species_count': 0,
            'accuracy_scores': [],
            'completeness_scores': [],
            'relevance_scores': []
        })
        
        self.routing_stats = {
            'total_routes': 0,
            'biomass_routes': 0,
            'other_routes': 0,
            'confusion_matrix': defaultdict(lambda: defaultdict(int))
        }
        
        self.error_count = 0
        self.empty_response_count = 0
    
    def load_components(self):
        """加载所有系统组件"""
        logger.info("=" * 80)
        logger.info("🔧 加载系统组件...")
        
        try:
            # 1. 加载Qwen微调模型
            logger.info("  [1/4] 加载Qwen微调模型...")
            from modules.qwen_model_adapter import QwenModelAdapter
            
            self.qwen_model = QwenModelAdapter(
                base_model_path=self.base_model_path,
                lora_path=self.lora_path
            )
            self.qwen_model.load_model()
            logger.info("    ✅ Qwen模型加载成功")
            
            # 2. 加载通用对话模块
            logger.info("  [2/4] 加载通用对话模块...")
            from modules.general_chat import GeneralChat
            
            self.general_chat = GeneralChat(self.qwen_model)
            logger.info("    ✅ 通用对话模块加载成功")
            
            # 3. 加载林业问答模块
            logger.info("  [3/4] 加载林业问答模块...")
            from modules.qa_module import ForestryQA
            
            self.forestry_qa = ForestryQA(self.qwen_model)
            logger.info("    ✅ 林业问答模块加载成功")
            
            # 4. ✅ 加载BiomassEstimatorBasic（严格匹配版本）
            logger.info("  [4/4] 加载生物量估算模块（基础版-严格匹配）...")
            try:
                from biomass_estimator_basic import BiomassEstimatorBasic
                
                self.biomass_estimator = BiomassEstimatorBasic(
                    equation_db_path=self.equation_db_path
                )
                logger.info("    ✅ 生物量估算模块加载成功（严格字符串匹配）")
            except Exception as e:
                logger.warning(f"    ⚠️ 生物量估算模块加载失败，将使用微调模型替代: {e}")
                self.biomass_estimator = None
            
            # 配置评估器
            self.metrics_calculator.set_qwen_model(self.qwen_model)
            logger.info("✅ 所有组件加载完成")
            
        except Exception as e:
            logger.error(f"❌ 组件加载失败: {e}", exc_info=True)
            sys.exit(1)
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """✅ 加载测试数据（使用新的加载器）"""
        logger.info("=" * 80)
        logger.info(f"📂 加载测试数据...")
        
        loader = TestDataLoader()
        
        # 加载生物量测试数据
        test_data = loader.load_biomass_test_data(self.biomass_test_file)
        
        if not test_data:
            logger.error(f"❌ 测试数据加载失败或为空")
            sys.exit(1)
        
        logger.info(f"✅ 成功加载 {len(test_data)} 条测试样本")
        
        # 统计
        task_counts = defaultdict(int)
        for data in test_data:
            task_type = data.get('task_type', 'unknown')
            task_counts[task_type] += 1
        
        logger.info("📊 任务分布:")
        for task, count in sorted(task_counts.items()):
            logger.info(f"  - {task}: {count} ({count/len(test_data)*100:.1f}%)")
        
        return test_data
    
    def route_query(self, instruction: str) -> Tuple[str, Dict[str, Any]]:
        """固定规则路由：基于关键词匹配"""
        route_start_time = time.time()
        
        is_biomass = self.keyword_matcher.is_biomass_query(instruction)
        matched_keywords = self.keyword_matcher.get_matched_keywords(instruction)
        
        if is_biomass:
            predicted_task = 'biomass'
            reason = f"关键词匹配: {', '.join(matched_keywords[:3])}"
            self.routing_stats['biomass_routes'] += 1
        else:
            predicted_task = 'general_chat'
            reason = "未匹配生物量关键词"
            self.routing_stats['other_routes'] += 1
        
        self.routing_stats['total_routes'] += 1
        routing_time = time.time() - route_start_time
        
        routing_info = {
            'predicted_task': predicted_task,
            'matched_keywords': matched_keywords,
            'reason': reason,
            'routing_time': routing_time,
            'method': 'keyword_matching'
        }
        
        return predicted_task, routing_info
    
    def generate_answer(self, instruction: str, predicted_task: str) -> Tuple[str, float]:
        """根据路由结果生成答案"""
        gen_start_time = time.time()
        predicted_answer = ""
        
        try:
            if predicted_task == 'biomass':
                if self.biomass_estimator is not None:
                    try:
                        result = self.biomass_estimator.estimate(instruction)
                        predicted_answer = result.get('answer', result.get('response', ''))
                    except Exception as e:
                        logger.warning(f"生物量估算模块失败，使用微调模型: {e}")
                        predicted_answer = self.qwen_model.generate(instruction)
                else:
                    predicted_answer = self.qwen_model.generate(instruction)
            
            else:
                predicted_answer = self.qwen_model.generate(instruction)
            
            if not predicted_answer:
                self.empty_response_count += 1
                logger.warning(f"任务 {predicted_task} 返回空响应")
            
        except Exception as e:
            logger.error(f"任务 {predicted_task} 生成失败: {e}")
            self.error_count += 1
            predicted_answer = ""
        
        generation_time = time.time() - gen_start_time
        
        return predicted_answer, generation_time
    
    def evaluate_single_sample(self, data: Dict[str, Any], idx: int, 
                              routing_log_handle) -> Dict[str, Any]:
        """评估单个样本"""
        true_task_type = data.get('task_type', 'unknown')
        instruction = data.get('instruction', '').strip()
        
        # ✅ 修改：获取参考答案（biomass需要字典格式）
        if true_task_type == 'biomass':
            reference = {
                'equation': data.get('equation', ''),
                'biomass': data.get('biomass', 0.0)
            }
        else:
            reference = data.get('output', '')
        
        if not instruction.strip():
            logger.warning(f"样本 {idx} instruction为空，跳过")
            return None
        
        start_time = time.time()
        
        # 1. 路由
        predicted_task, routing_info = self.route_query(instruction)
        
        # 2. 生成答案
        predicted_answer, generation_time = self.generate_answer(instruction, predicted_task)
        
        total_processing_time = time.time() - start_time
        
        # 3. 评估路由
        route_correct = (predicted_task == true_task_type)
        self.routing_stats['confusion_matrix'][true_task_type][predicted_task] += 1
        
        # 4. ✅ 评估答案（使用LLM增强评估）
        if true_task_type == 'biomass':
            metrics = self.metrics_calculator.calculate_biomass_metrics(
                predicted_answer=predicted_answer,
                reference=reference,
                instruction=instruction,
                equation_db=getattr(self.biomass_estimator, 'equation_db', None) if self.biomass_estimator else None
            )
        else:
            metrics = self.metrics_calculator.calculate_qa_metrics(
                predicted=predicted_answer,
                reference=reference,
                question=instruction,
                task_type=true_task_type
            )
        
        # 5. 更新统计
        self.task_stats[true_task_type]['total'] += 1
        self.task_stats[true_task_type]['route_correct'] += int(route_correct)
        self.task_stats[true_task_type]['answer_correct'] += metrics['score']
        self.task_stats[true_task_type]['scores'].append(metrics['score'])
        self.task_stats[true_task_type]['response_lengths'].append(len(predicted_answer))
        self.task_stats[true_task_type]['processing_times'].append(total_processing_time)
        self.task_stats[true_task_type]['routing_times'].append(routing_info['routing_time'])
        self.task_stats[true_task_type]['generation_times'].append(generation_time)
        
        # 记录详细维度
        if true_task_type == 'biomass':
            details = metrics.get('details', {})
            self.task_stats[true_task_type]['species_scores'].append(details.get('species', 0))
            self.task_stats[true_task_type]['region_scores'].append(details.get('region', 0))
            self.task_stats[true_task_type]['component_scores'].append(details.get('component', 0))
            self.task_stats[true_task_type]['variable_scores'].append(details.get('variables', 0))
            self.task_stats[true_task_type]['form_scores'].append(details.get('form', 0))
            self.task_stats[true_task_type]['coefficient_scores'].append(details.get('coefficient', 0))
            
            if details.get('used_fallback_region'):
                self.task_stats[true_task_type]['fallback_region_count'] += 1
            if details.get('used_fallback_species'):
                self.task_stats[true_task_type]['fallback_species_count'] += 1
                
        elif true_task_type in ['forestry_qa', 'general_chat']:
            self.task_stats[true_task_type]['accuracy_scores'].append(metrics.get('accuracy', 0))
            self.task_stats[true_task_type]['completeness_scores'].append(metrics.get('completeness', 0))
            self.task_stats[true_task_type]['relevance_scores'].append(metrics.get('relevance', 0))
        
        # 6. 记录路由日志
        routing_log = {
            'index': idx,
            'true_task': true_task_type,
            'predicted_task': predicted_task,
            'route_correct': route_correct,
            'routing_info': routing_info
        }
        routing_log_handle.write(json.dumps(routing_log, ensure_ascii=False) + '\n')
        
        # 7. 构建结果
        result = {
            'index': idx,
            'true_task_type': true_task_type,
            'predicted_task_type': predicted_task,
            'instruction': instruction,
            'reference': reference,
            'predicted_answer': predicted_answer,
            'routing': {
                'route_correct': route_correct,
                'predicted_task': predicted_task,
                'matched_keywords': routing_info['matched_keywords'],
                'reason': routing_info['reason'],
                'method': routing_info['method'],
                'routing_time': routing_info['routing_time']
            },
            'metrics': metrics,
            'score': metrics['score'],
            'timing': {
                'total_time': total_processing_time,
                'routing_time': routing_info['routing_time'],
                'generation_time': generation_time
            },
            'response_length': len(predicted_answer),
            'metadata': data.get('metadata', {})
        }
        
        return result
    
    def run_evaluation(self):
        """运行完整评估"""
        logger.info("=" * 80)
        logger.info("🚀 开始E2实验评估...")
        logger.info(f"  实验: 固定规则路由 + 严格字符串匹配")
        logger.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_data = self.load_test_data()
        total_samples = len(test_data)
        
        detailed_file_handle = open(self.detailed_file, 'w', encoding='utf-8')
        routing_log_handle = open(self.routing_log_file, 'w', encoding='utf-8')
        
        start_time = time.time()
        
        for idx, data in enumerate(test_data):
            result = self.evaluate_single_sample(data, idx, routing_log_handle)
            
            if result:
                self.results.append(result)
                detailed_file_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (idx + 1) / elapsed
                eta_seconds = (total_samples - idx - 1) / samples_per_sec if samples_per_sec > 0 else 0
                
                total_evaluated = sum(s['total'] for s in self.task_stats.values())
                total_correct = sum(s['answer_correct'] for s in self.task_stats.values())
                total_route_correct = sum(s['route_correct'] for s in self.task_stats.values())
                
                current_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0
                current_route_accuracy = total_route_correct / total_evaluated if total_evaluated > 0 else 0.0
                
                logger.info(
                    f"  进度: {idx + 1}/{total_samples} ({(idx+1)/total_samples*100:.1f}%) | "
                    f"答案得分: {current_accuracy:.2%} | "
                    f"路由准确率: {current_route_accuracy:.2%} | "
                    f"速度: {samples_per_sec:.2f} 样本/秒 | "
                    f"预计剩余: {eta_seconds/60:.1f} 分钟"
                )
        
        detailed_file_handle.close()
        routing_log_handle.close()
        
        total_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"✅ 评估完成! 总耗时: {total_time/60:.1f} 分钟")
        logger.info(f"  处理样本数: {len(self.results)}/{total_samples}")
        logger.info(f"  错误数: {self.error_count}")
        logger.info(f"  空响应数: {self.empty_response_count}")
    
    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """计算汇总指标"""
        logger.info("=" * 80)
        logger.info("📊 计算汇总指标...")
        
        summary = {
            'experiment_name': 'E2-Fixed-Rule-Routing',
            'model_type': 'QwenLoRA_KeywordRouting_StrictMatch',
            'biomass_estimator': 'BiomassEstimatorBasic (严格字符串匹配)',
            'evaluation_method': 'LLM_Enhanced_Unified',
            'total_samples': len(self.results),
            'error_count': self.error_count,
            'empty_response_count': self.empty_response_count,
            'routing_statistics': {
                'total_routes': self.routing_stats['total_routes'],
                'biomass_routes': self.routing_stats['biomass_routes'],
                'other_routes': self.routing_stats['other_routes'],
            },
            'confusion_matrix': dict(self.routing_stats['confusion_matrix']),
            'task_performance': {}
        }
        
        for task_type, stats in self.task_stats.items():
            if stats['total'] == 0:
                continue
            
            task_metrics = {
                'total_samples': stats['total'],
                'route_accuracy': stats['route_correct'] / stats['total'],
                'answer_accuracy': stats['answer_correct'] / stats['total'],
                'average_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                'std_score': np.std(stats['scores']) if stats['scores'] else 0.0,
                'average_total_time': np.mean(stats['processing_times']) if stats['processing_times'] else 0.0,
            }
            
            if task_type == 'biomass':
                task_metrics.update({
                    'average_species_match': np.mean(stats['species_scores']) if stats['species_scores'] else 0.0,
                    'average_region_match': np.mean(stats['region_scores']) if stats['region_scores'] else 0.0,
                    'average_component_match': np.mean(stats['component_scores']) if stats['component_scores'] else 0.0,
                    'average_variable_match': np.mean(stats['variable_scores']) if stats['variable_scores'] else 0.0,
                    'average_form_match': np.mean(stats['form_scores']) if stats['form_scores'] else 0.0,
                    'average_coefficient_similarity': np.mean(stats['coefficient_scores']) if stats['coefficient_scores'] else 0.0,
                })
                
                logger.info(f"\n📈 {task_type} 性能:")
                logger.info(f"  - 样本数: {task_metrics['total_samples']}")
                logger.info(f"  - 路由准确率: {task_metrics['route_accuracy']:.2%}")
                logger.info(f"  - 答案准确率: {task_metrics['answer_accuracy']:.2%}")
                logger.info(f"  - 平均得分: {task_metrics['average_score']:.2f} ± {task_metrics['std_score']:.2f}")
                logger.info(f"  📊 多维度匹配:")
                logger.info(f"    - 树种: {task_metrics['average_species_match']:.2f}")
                logger.info(f"    - 地区: {task_metrics['average_region_match']:.2f}")
                logger.info(f"    - 组分: {task_metrics['average_component_match']:.2f}")
            
            summary['task_performance'][task_type] = task_metrics
        
        total_samples = sum(s['total'] for s in self.task_stats.values())
        
        if total_samples > 0:
            overall_metrics = {
                'route_accuracy': sum(s['route_correct'] for s in self.task_stats.values()) / total_samples,
                'answer_accuracy': sum(s['answer_correct'] for s in self.task_stats.values()) / total_samples,
                'average_score': np.mean([score for s in self.task_stats.values() for score in s['scores']]),
            }
            
            summary['overall_performance'] = overall_metrics
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 整体性能:")
            logger.info(f"  - 总样本数: {total_samples}")
            logger.info(f"  - 路由准确率: {overall_metrics['route_accuracy']:.2%}")
            logger.info(f"  - 答案准确率: {overall_metrics['answer_accuracy']:.2%}")
            logger.info(f"  - 平均得分: {overall_metrics['average_score']:.2f}")
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """保存结果"""
        logger.info("=" * 80)
        logger.info(f"💾 保存结果到: {self.output_file}")
        
        full_results = {
            'metadata': {
                'experiment': 'E2-Fixed-Rule-Routing',
                'timestamp': datetime.now().isoformat(),
                'base_model': self.base_model_path,
                'lora_model': self.lora_path,
                'equation_db': self.equation_db_path,
                'test_file': self.biomass_test_file,
                'evaluation_method': 'LLM_Enhanced_Unified'
            },
            'summary': summary,
            'sample_results': self.results[:200]
        }
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 汇总结果已保存: {self.output_file}")
            logger.info(f"✅ 详细结果已保存: {self.detailed_file}")
            logger.info(f"✅ 路由日志已保存: {self.routing_log_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}", exc_info=True)
    
    def run(self):
        """运行完整的E2实验"""
        logger.info("🎯 开始执行 E2-Fixed-Rule-Routing 实验")
        logger.info(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"✅ 关键特点: 严格字符串匹配（无同义词泛化）")
        logger.info(f"✅ 预期得分: 0.50-0.60")
        
        self.load_components()
        self.run_evaluation()
        summary = self.calculate_summary_metrics()
        self.save_results(summary)
        
        logger.info("=" * 80)
        logger.info("🎉 E2-Fixed-Rule-Routing 实验完成!")
        logger.info("=" * 80)


if __name__ == "__main__":
    experiment = FixedRuleRoutingExperiment()
    experiment.run()