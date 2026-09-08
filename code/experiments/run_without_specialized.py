import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版：主实验E3 - Router System with LLM Enhanced Evaluation
集成LLM增强评估（多维度评分 + LLM兜底）
功能：测试带路由器的完整系统在所有任务上的性能，使用语义理解评分

修复内容：
- ✅ 修复第295行：general_chat.chat() → general_chat.process()

日期：2026-05-25（修复版）
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
sys.path.insert(0, '/data/nefu/毕业论文实验')

# 导入LLM增强评估模块
from metrics_llm_enhanced import LLMEnhancedMetricsCalculator

# 配置日志
log_dir = "/data/nefu/毕业论文实验/experiments/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'E3_llm_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class E3LLMEnhancedExperiment:
    """E3实验主类 - LLM增强评分系统"""
    
    def __init__(self):
        # 路径配置
        self.base_model_path = "/data/nefu/homee/Qwen/Qwen1.5-7B-Chat"
        self.lora_path = "/data/nefu/毕业论文实验/checkpoints/Qwen1.5-7B-Chat_lora/checkpoint-74170"
        self.test_file = "/data/nefu/毕业论文实验/experiments/results/test_all_tasks.jsonl"
        
        # DeepSeek API配置
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 输出配置
        self.output_dir = "/data/nefu/毕业论文实验/experiments/results/main"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.output_file = os.path.join(self.output_dir, "E3_no_specialized_module_results.json")
        self.detailed_file = os.path.join(self.output_dir, "E3_no_specialized_module_detailed.jsonl")
        self.routing_log_file = os.path.join(self.output_dir, "E3_no_specialized_module_routing_log.jsonl")
        
        # 组件
        self.router = None
        self.qwen_model = None
        self.general_chat = None
        self.forestry_qa = None
        self.biomass_estimator = None
        
        # 初始化结果列表
        self.results = []
        
        # 初始化LLM增强评估器
        self.metrics_calculator = LLMEnhancedMetricsCalculator(
            deepseek_api_key=self.deepseek_api_key,
            qwen_model=None  # 稍后在load_components中赋值
        )
        
        # 任务映射
        self.task_map = {
            'biomass': 'biomass',
            'forestry': 'forestry_qa',
            'general': 'general_chat'
        }
        
        self.reverse_task_map = {
            'biomass': 'biomass',
            'forestry_qa': 'forestry',
            'general_chat': 'general'
        }
        
        # 统计字段
        self.task_stats = defaultdict(lambda: {
            'total': 0,
            'route_correct': 0,
            'answer_correct': 0,
            'scores': [],
            'response_lengths': [],
            'processing_times': [],
            'routing_times': [],
            'generation_times': [],
            # Biomass详细维度
            'species_scores': [],
            'region_scores': [],
            'component_scores': [],
            'variable_scores': [],
            'form_scores': [],
            'coefficient_scores': [],
            'fallback_region_count': 0,
            'fallback_species_count': 0,
            # Forestry/General详细维度
            'accuracy_scores': [],
            'completeness_scores': [],
            'relevance_scores': []
        })
        
        # 路由统计
        self.routing_stats = {
            'total_routes': 0,
            'rule_based_routes': 0,
            'llm_based_routes': 0,
            'deepseek_routing_calls': 0,
            'routing_errors': 0,
            'confusion_matrix': defaultdict(lambda: defaultdict(int))
        }
        
        self.error_count = 0
        self.empty_response_count = 0
    
    def load_components(self):
        """加载所有系统组件"""
        logger.info("=" * 80)
        logger.info("🔧 加载系统组件...")
        
        try:
            # 1. 加载路由器
            logger.info("  [1/6] 加载智能路由器...")
            from intelligent_router import IntelligentRouter
            
            self.router = IntelligentRouter(deepseek_api_key=self.deepseek_api_key)
            logger.info("    ✅ 路由器加载成功")
            
            # 2. 加载Qwen微调模型
            logger.info("  [2/6] 加载Qwen微调模型...")
            from modules.qwen_model_adapter import QwenModelAdapter
            
            self.qwen_model = QwenModelAdapter(
                base_model_path=self.base_model_path,
                lora_path=self.lora_path
            )
            self.qwen_model.load_model()
            logger.info("    ✅ Qwen模型加载成功")
            
            # 3. 加载通用对话模块
            logger.info("  [3/6] 加载通用对话模块...")
            from modules.general_chat import GeneralChat
            
            self.general_chat = GeneralChat(self.qwen_model)
            logger.info("    ✅ 通用对话模块加载成功")
            
            # 4. 加载林业问答模块
            logger.info("  [4/6] 加载林业问答模块...")
            from modules.qa_module import ForestryQA
            
            self.forestry_qa = ForestryQA(self.qwen_model)
            logger.info("    ✅ 林业问答模块加载成功")
            
            # 5. 加载生物量估算模块
            logger.info("  [5/6] 尝试加载生物量估算模块...")
            try:
                from biomass_estimator_enhanced_v2 import BiomassEstimatorEnhancedV2
                
                self.biomass_estimator = BiomassEstimatorEnhancedV2()
                logger.info("    ✅ 生物量估算模块加载成功")
            except Exception as e:
                logger.warning(f"    ⚠️ 生物量估算模块加载失败，将使用微调模型替代: {e}")
                self.biomass_estimator = None
            
            # 6. 配置LLM增强评估器
            logger.info("  [6/6] 配置LLM增强评估器...")
            self.metrics_calculator.set_qwen_model(self.qwen_model)
            logger.info("    ✅ 评估器配置完成（支持LLM地学/生态学兜底）")
            
            logger.info("✅ 所有组件加载完成")
            
        except ImportError as e:
            logger.error(f"❌ 导入组件失败: {e}")
            logger.error("请确保所有必需的模块都存在")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ 组件加载失败: {e}", exc_info=True)
            sys.exit(1)
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        logger.info("=" * 80)
        logger.info(f"📂 加载测试数据: {self.test_file}")
        
        if not os.path.exists(self.test_file):
            logger.error(f"❌ 测试文件不存在: {self.test_file}")
            logger.info("💡 请先运行 prepare_unified_test_sets.py 生成测试集")
            sys.exit(1)
        
        test_data = []
        
        try:
            with open(self.test_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        test_data.append(data)
                    except json.JSONDecodeError:
                        logger.warning(f"行 {idx + 1} JSON解析失败，跳过")
                        continue
            
            logger.info(f"✅ 成功加载 {len(test_data)} 条测试样本")
            
            # 统计各任务数量
            task_counts = defaultdict(int)
            for data in test_data:
                task_type = data.get('task_type', 'unknown')
                task_counts[task_type] += 1
            
            logger.info("📊 任务分布:")
            for task, count in sorted(task_counts.items()):
                logger.info(f"  - {task}: {count} ({count/len(test_data)*100:.1f}%)")
            
            return test_data
            
        except Exception as e:
            logger.error(f"❌ 加载测试数据失败: {e}", exc_info=True)
            sys.exit(1)
    
    def route_query(self, instruction: str) -> Tuple[str, Dict[str, Any]]:
        """路由查询到对应模块"""
        route_start_time = time.time()
        
        try:
            route_result = self.router.route(instruction)
            routing_time = time.time() - route_start_time
            
            predicted_module = route_result.get('module', 'unknown')
            confidence = route_result.get('confidence', 0.0)
            reason = route_result.get('reason', '')
            
            # 统计路由类型
            self.routing_stats['total_routes'] += 1
            
            if 'LLM' in reason or 'DeepSeek' in reason:
                self.routing_stats['llm_based_routes'] += 1
                self.routing_stats['deepseek_routing_calls'] += 1
            else:
                self.routing_stats['rule_based_routes'] += 1
            
            routing_info = {
                'predicted_module': predicted_module,
                'confidence': confidence,
                'reason': reason,
                'routing_time': routing_time,
                'method': 'llm' if 'LLM' in reason else 'rule'
            }
            
            return predicted_module, routing_info
            
        except Exception as e:
            logger.error(f"路由失败: {e}")
            self.routing_stats['routing_errors'] += 1
            
            routing_info = {
                'predicted_module': 'general',
                'confidence': 0.0,
                'reason': f'Routing error: {str(e)}',
                'routing_time': time.time() - route_start_time,
                'method': 'error_fallback'
            }
            
            return 'general', routing_info
    
    def generate_answer(self, instruction: str, predicted_module: str) -> Tuple[str, float]:
        """根据路由结果生成答案"""
        gen_start_time = time.time()
        predicted_answer = ""
        
        try:
            if predicted_module == 'biomass':
                if self.biomass_estimator is not None:
                    try:
                        result = {
    "answer":
    self.qwen_model.generate(instruction)
}
                        predicted_answer = result.get('answer', result.get('response', ''))
                    except Exception as e:
                        logger.warning(f"生物量估算模块失败，使用微调模型: {e}")
                        predicted_answer = self.qwen_model.generate(instruction)
                else:
                    predicted_answer = self.qwen_model.generate(instruction)
            
            elif predicted_module == 'forestry':
                result = self.forestry_qa.process(instruction)

                if isinstance(result, dict):
                    predicted_answer = result.get('answer', '')
                else:
                    predicted_answer = result
            
            else:  # general
                # ✅ 修复：调用process方法而非不存在的chat方法
                result = self.general_chat.process(instruction)
                predicted_answer = result.get('answer', '')
            
            if not predicted_answer:
                self.empty_response_count += 1
                logger.warning(f"模块 {predicted_module} 返回空响应")
            
        except Exception as e:
            logger.error(f"模块 {predicted_module} 生成失败: {e}")
            self.error_count += 1
            predicted_answer = ""
        
        generation_time = time.time() - gen_start_time
        
        return predicted_answer, generation_time
    
    def evaluate_single_sample(self, data: Dict[str, Any], idx: int, 
                              routing_log_handle) -> Dict[str, Any]:
        """评估单个样本"""
        true_task_type = data.get('task_type', 'unknown')
        instruction = data.get('instruction', '').strip()
        
        # 获取参考答案
        if true_task_type == 'biomass':
            reference = data.get('equation', '')
        else:
            reference = data.get('output', '')
        
        if not instruction.strip():
            logger.warning(f"样本 {idx} instruction为空，跳过")
            return None
        
        start_time = time.time()
        
        # 1. 路由查询
        predicted_module, routing_info = self.route_query(instruction)
        predicted_task_type = self.task_map.get(predicted_module, predicted_module)
        
        # 2. 生成答案
        predicted_answer, generation_time = self.generate_answer(instruction, predicted_module)
        
        total_processing_time = time.time() - start_time
        
        # 3. 评估路由准确性
        route_correct = (predicted_task_type == true_task_type)
        
        # 更新混淆矩阵
        self.routing_stats['confusion_matrix'][true_task_type][predicted_task_type] += 1
        
        # 4. 评估答案质量
        if true_task_type == 'biomass':
            # Biomass：多维度评分 + LLM兜底
            metrics = self.metrics_calculator.calculate_biomass_metrics(
                predicted_answer=predicted_answer,
                reference=reference,
                instruction=instruction,
                equation_db=getattr(self.biomass_estimator, 'equation_db', None)
            )
        else:
            # Forestry/General：准确性、完整性、相关性
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
            # Biomass多维度得分
            details = metrics.get('details', {})
            self.task_stats[true_task_type]['species_scores'].append(details.get('species', 0))
            self.task_stats[true_task_type]['region_scores'].append(details.get('region', 0))
            self.task_stats[true_task_type]['component_scores'].append(details.get('component', 0))
            self.task_stats[true_task_type]['variable_scores'].append(details.get('variables', 0))
            self.task_stats[true_task_type]['form_scores'].append(details.get('form', 0))
            self.task_stats[true_task_type]['coefficient_scores'].append(details.get('coefficient', 0))
            
            # 兜底策略统计
            if details.get('used_fallback_region'):
                self.task_stats[true_task_type]['fallback_region_count'] += 1
            if details.get('used_fallback_species'):
                self.task_stats[true_task_type]['fallback_species_count'] += 1

        elif true_task_type in ['forestry_qa', 'general_chat']:
            # Forestry/General三维度得分
            self.task_stats[true_task_type]['accuracy_scores'].append(metrics.get('accuracy', 0))
            self.task_stats[true_task_type]['completeness_scores'].append(metrics.get('completeness', 0))
            self.task_stats[true_task_type]['relevance_scores'].append(metrics.get('relevance', 0))
        
        # 6. 记录路由日志
        routing_log = {
            'index': idx,
            'true_task': true_task_type,
            'predicted_task': predicted_task_type,
            'route_correct': route_correct,
            'routing_info': routing_info
        }
        routing_log_handle.write(json.dumps(routing_log, ensure_ascii=False) + '\n')
        
        # 7. 构建结果
        result = {
            'index': idx,
            'true_task_type': true_task_type,
            'predicted_task_type': predicted_task_type,
            'instruction': instruction,
            'reference': reference,
            'predicted_answer': predicted_answer,
            'routing': {
                'route_correct': route_correct,
                'predicted_module': predicted_module,
                'confidence': routing_info['confidence'],
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
        logger.info("🚀 开始E3实验评估...")
        logger.info(f"  实验: LLM增强评分系统（语义理解）")
        logger.info(f"  评估方法: 多维度评分 + LLM兜底")
        logger.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 加载测试数据
        test_data = self.load_test_data()
        total_samples = len(test_data)
        
        # 打开文件句柄
        detailed_file_handle = open(self.detailed_file, 'w', encoding='utf-8')
        routing_log_handle = open(self.routing_log_file, 'w', encoding='utf-8')
        
        start_time = time.time()
        
        # 逐样本评估
        for idx, data in enumerate(test_data):
            result = self.evaluate_single_sample(data, idx, routing_log_handle)
            
            if result:
                self.results.append(result)
                detailed_file_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 定期输出进度
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (idx + 1) / elapsed
                eta_seconds = (total_samples - idx - 1) / samples_per_sec if samples_per_sec > 0 else 0
                
                # 计算当前指标
                total_evaluated = sum(s['total'] for s in self.task_stats.values())
                total_correct = sum(s['answer_correct'] for s in self.task_stats.values())
                total_route_correct = sum(s['route_correct'] for s in self.task_stats.values())
                
                current_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0
                current_route_accuracy = total_route_correct / total_evaluated if total_evaluated > 0 else 0.0
                
                logger.info(
                    f"  进度: {idx + 1}/{total_samples} ({(idx+1)/total_samples*100:.1f}%) | "
                    f"答案准确率: {current_accuracy:.2%} | "
                    f"路由准确率: {current_route_accuracy:.2%} | "
                    f"DeepSeek路由: {self.routing_stats['deepseek_routing_calls']} | "
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
        logger.info(f"  DeepSeek路由调用: {self.routing_stats['deepseek_routing_calls']} 次")
    
    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """计算汇总指标"""
        logger.info("=" * 80)
        logger.info("📊 计算汇总指标...")
        
        summary = {
            'experiment_name': 'E3-LLM-Enhanced',
            'model_type': 'QwenLoRA_WithRouter_LLMEval',
            'evaluation_method': 'Multidimensional_LLM_Fallback',
            'total_samples': len(self.results),
            'error_count': self.error_count,
            'empty_response_count': self.empty_response_count,
            'routing_statistics': {
                'total_routes': self.routing_stats['total_routes'],
                'rule_based_routes': self.routing_stats['rule_based_routes'],
                'llm_based_routes': self.routing_stats['llm_based_routes'],
                'deepseek_routing_calls': self.routing_stats['deepseek_routing_calls'],
                'routing_errors': self.routing_stats['routing_errors'],
                'rule_based_percentage': self.routing_stats['rule_based_routes'] / self.routing_stats['total_routes'] * 100 if self.routing_stats['total_routes'] > 0 else 0,
                'llm_based_percentage': self.routing_stats['llm_based_routes'] / self.routing_stats['total_routes'] * 100 if self.routing_stats['total_routes'] > 0 else 0
            },
            'confusion_matrix': dict(self.routing_stats['confusion_matrix']),
            'task_performance': {}
        }
        
        # 计算每个任务的详细指标
        for task_type, stats in self.task_stats.items():
            if stats['total'] == 0:
                continue
            
            task_metrics = {
                'total_samples': stats['total'],
                'route_accuracy': stats['route_correct'] / stats['total'],
                'answer_accuracy': stats['answer_correct'] / stats['total'],
                'average_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                'std_score': np.std(stats['scores']) if stats['scores'] else 0.0,
                'min_score': np.min(stats['scores']) if stats['scores'] else 0.0,
                'max_score': np.max(stats['scores']) if stats['scores'] else 0.0,
                'average_response_length': np.mean(stats['response_lengths']) if stats['response_lengths'] else 0.0,
                'average_total_time': np.mean(stats['processing_times']) if stats['processing_times'] else 0.0,
                'average_routing_time': np.mean(stats['routing_times']) if stats['routing_times'] else 0.0,
                'average_generation_time': np.mean(stats['generation_times']) if stats['generation_times'] else 0.0
            }
            
            if task_type == 'biomass':
                task_metrics.update({
                    'average_species_match': np.mean(stats['species_scores']) if stats['species_scores'] else 0.0,
                    'average_region_match': np.mean(stats['region_scores']) if stats['region_scores'] else 0.0,
                    'average_component_match': np.mean(stats['component_scores']) if stats['component_scores'] else 0.0,
                    'average_variable_match': np.mean(stats['variable_scores']) if stats['variable_scores'] else 0.0,
                    'average_form_match': np.mean(stats['form_scores']) if stats['form_scores'] else 0.0,
                    'average_coefficient_similarity': np.mean(stats['coefficient_scores']) if stats['coefficient_scores'] else 0.0,
                    'fallback_region_count': stats['fallback_region_count'],
                    'fallback_species_count': stats['fallback_species_count'],
                    'fallback_region_rate': stats['fallback_region_count'] / stats['total'] if stats['total'] > 0 else 0.0,
                    'fallback_species_rate': stats['fallback_species_count'] / stats['total'] if stats['total'] > 0 else 0.0
                })
                
                logger.info(f"\n📈 {task_type} 性能:")
                logger.info(f"  - 样本数: {task_metrics['total_samples']}")
                logger.info(f"  - 路由准确率: {task_metrics['route_accuracy']:.2%}")
                logger.info(f"  - 答案准确率: {task_metrics['answer_accuracy']:.2%}")
                logger.info(f"  - 平均得分: {task_metrics['average_score']:.2f} ± {task_metrics['std_score']:.2f}")
                logger.info(f"  📊 详细维度:")
                logger.info(f"    - 准确性: {task_metrics.get('average_accuracy', 0):.1f}/10")
                logger.info(f"    - 完整性: {task_metrics.get('average_completeness', 0):.1f}/10")
                logger.info(f"    - 相关性: {task_metrics.get('average_relevance', 0):.1f}/10")
            
            summary['task_performance'][task_type] = task_metrics
        
        # 计算整体指标
        total_samples = sum(s['total'] for s in self.task_stats.values())
        
        if total_samples > 0:
            overall_metrics = {
                'route_accuracy': sum(s['route_correct'] for s in self.task_stats.values()) / total_samples,
                'answer_accuracy': sum(s['answer_correct'] for s in self.task_stats.values()) / total_samples,
                'average_score': np.mean([score for s in self.task_stats.values() for score in s['scores']]),
                'average_total_time': np.mean([t for s in self.task_stats.values() for t in s['processing_times']]),
                'average_routing_time': np.mean([t for s in self.task_stats.values() for t in s['routing_times']]),
                'average_generation_time': np.mean([t for s in self.task_stats.values() for t in s['generation_times']])
            }
            
            summary['overall_performance'] = overall_metrics
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 整体性能:")
            logger.info(f"  - 总样本数: {total_samples}")
            logger.info(f"  - 路由准确率: {overall_metrics['route_accuracy']:.2%}")
            logger.info(f"  - 答案准确率: {overall_metrics['answer_accuracy']:.2%}")
            logger.info(f"  - 平均得分: {overall_metrics['average_score']:.2f}")
            logger.info(f"  - 平均总耗时: {overall_metrics['average_total_time']:.3f} 秒/样本")
            
            # 打印混淆矩阵
            logger.info("\n📋 路由混淆矩阵:")
            for true_task in sorted(self.routing_stats['confusion_matrix'].keys()):
                logger.info(f"  {true_task}:")
                for pred_task, count in sorted(self.routing_stats['confusion_matrix'][true_task].items()):
                    percentage = count / self.task_stats[true_task]['total'] * 100 if self.task_stats[true_task]['total'] > 0 else 0
                    logger.info(f"    → {pred_task}: {count} ({percentage:.1f}%)")
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """保存结果"""
        logger.info("=" * 80)
        logger.info(f"💾 保存结果到: {self.output_file}")
        
        full_results = {
            'metadata': {
                'experiment': 'E3-LLM-Enhanced-FIXED',
                'timestamp': datetime.now().isoformat(),
                'base_model': self.base_model_path,
                'lora_model': self.lora_path,
                'test_file': self.test_file,
                'evaluation_method': 'Multidimensional_LLM_Fallback',
                'fix_version': 'v1.0 - general_chat.process() fix'
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
        """运行完整的E3实验"""
        logger.info("🎯 开始执行 E3-LLM-Enhanced (修复版) 实验")
        logger.info(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"修复内容: general_chat.chat() → general_chat.process()")
        
        # 1. 加载所有组件
        self.load_components()
        
        # 2. 运行评估
        self.run_evaluation()
        
        # 3. 计算汇总指标
        summary = self.calculate_summary_metrics()
        
        # 4. 保存结果
        self.save_results(summary)
        
        logger.info("=" * 80)
        logger.info("🎉 E3-LLM-Enhanced 实验完成!")
        logger.info("=" * 80)


if __name__ == "__main__":
    experiment = E3LLMEnhancedExperiment()
    experiment.run()