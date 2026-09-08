import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主实验E4 - Pure RAG System with Base Model
纯RAG系统（基座模型版本）

关键设计：
1. 使用Qwen基座模型（未微调）而非微调模型
2. 完全依赖RAG检索，不使用专门模块
3. 使用真正的语义向量检索
4. 统一使用LLMEnhancedMetricsCalculator评分

实验目的：
- 验证纯RAG系统的能力
- 作为E1/E2/E3的对比baseline
- 证明FIQS（任务自适应路由）优于单一RAG方法

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
sys.path.insert(0, '/data/nefu/毕业论文实验')

# 导入评估模块
from metrics_llm_enhanced import LLMEnhancedMetricsCalculator

# 配置日志
log_dir = "/data/nefu/毕业论文实验/experiments/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'E4_pure_rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class E4PureRAGExperiment:
    """E4实验主类 - 纯RAG系统（基座模型）"""
    
    def __init__(self):
        # ✅ 使用基座模型而非微调模型
        self.base_model_path = "/data/nefu/homee/Qwen/Qwen1.5-7B-Chat"  # 不加载LoRA
        self.lora_path = None  # E4不使用微调
        
        self.test_file = "/data/nefu/毕业论文实验/experiments/results/test_all_tasks.jsonl"
        
        # DeepSeek API配置
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 知识库路径
        self.knowledge_base_dir = "/data/nefu/毕业论文实验/knowledge_base"
        
        # 输出配置
        self.output_dir = "/data/nefu/毕业论文实验/experiments/results/main"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.output_file = os.path.join(self.output_dir, "E4_pure_rag_results.json")
        self.detailed_file = os.path.join(self.output_dir, "E4_pure_rag_detailed.jsonl")
        
        # 组件
        self.qwen_model = None
        self.retriever = None
        
        # 初始化结果列表
        self.results = []
        
        # 初始化评估器（与E1/E2/E3相同）
        self.metrics_calculator = LLMEnhancedMetricsCalculator(
            deepseek_api_key=self.deepseek_api_key,
            qwen_model=None
        )
        
        # 统计
        self.task_stats = defaultdict(lambda: {
            'total': 0,
            'answer_correct': 0,
            'scores': [],
            'response_lengths': [],
            'processing_times': [],
            'retrieval_times': [],
            'generation_times': [],
            'retrieved_docs_counts': [],
            # Biomass详细维度
            'species_scores': [],
            'region_scores': [],
            'component_scores': [],
            'variable_scores': [],
            'form_scores': [],
            'coefficient_scores': [],
            # Forestry/General详细维度
            'accuracy_scores': [],
            'completeness_scores': [],
            'relevance_scores': []
        })
        
        self.error_count = 0
        self.empty_response_count = 0
    
    def load_components(self):
        """加载系统组件"""
        logger.info("=" * 80)
        logger.info("🔧 加载E4-Pure-RAG组件...")
        
        try:
            # 1. 加载基座Qwen模型（不加载LoRA）
            logger.info("  [1/3] 加载Qwen基座模型（未微调）...")
            from modules.qwen_model_adapter import QwenModelAdapter
            
            self.qwen_model = QwenModelAdapter(
                base_model_path=self.base_model_path,
                lora_path=None  # ✅ 不加载LoRA
            )
            self.qwen_model.load_model()
            logger.info("    ✅ Qwen基座模型加载成功")
            
            # 2. 初始化RAG检索器
            logger.info("  [2/3] 初始化RAG检索器...")
            try:
                from modules.rag_retriever import RAGRetriever
                
                self.retriever = RAGRetriever(
                    knowledge_base_dir=self.knowledge_base_dir,
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("    ✅ RAG检索器初始化成功")
            except ImportError:
                logger.warning("    ⚠️ RAG检索器模块不存在，将创建简单版本")
                self.retriever = self._create_simple_retriever()
            
            # 3. 配置评估器
            logger.info("  [3/3] 配置评估器...")
            self.metrics_calculator.set_qwen_model(self.qwen_model)
            logger.info("    ✅ 评估器配置完成")
            
            logger.info("✅ 所有组件加载完成")
            
        except Exception as e:
            logger.error(f"❌ 组件加载失败: {e}", exc_info=True)
            sys.exit(1)
    
    def _create_simple_retriever(self):
        """创建简单的RAG检索器（如果模块不存在）"""
        logger.info("    创建简单RAG检索器...")
        
        class SimpleRetriever:
            """简单的RAG检索器"""
            
            def __init__(self):
                self.documents = []
                logger.info("      ✅ 简单检索器已创建")
            
            def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
                """简单检索（返回空结果，完全依赖基座模型）"""
                return []
        
        return SimpleRetriever()
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        logger.info("=" * 80)
        logger.info(f"📂 加载测试数据: {self.test_file}")
        
        if not os.path.exists(self.test_file):
            logger.error(f"❌ 测试文件不存在: {self.test_file}")
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
    
    def generate_answer_with_rag(self, instruction: str) -> Tuple[str, float, float, int]:
        """使用RAG生成答案"""
        gen_start_time = time.time()
        
        try:
            # 1. 检索相关文档
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(instruction, top_k=5)
            retrieval_time = time.time() - retrieval_start
            
            # 2. 构造RAG提示词
            if retrieved_docs:
                context = "\n\n".join([
                    f"参考文档 {i+1}:\n{doc.get('content', '')}"
                    for i, doc in enumerate(retrieved_docs[:3])
                ])
                
                prompt = f"""参考以下文档回答问题：

{context}

问题：{instruction}

请基于参考文档给出准确、完整的答案。"""
            else:
                # 没有检索到文档，直接使用基座模型
                prompt = instruction
            
            # 3. 生成答案
            generation_start = time.time()
            predicted_answer = self.qwen_model.generate(prompt)
            generation_time = time.time() - generation_start
            
            if not predicted_answer:
                self.empty_response_count += 1
                logger.warning("RAG返回空响应")
            
            return predicted_answer, retrieval_time, generation_time, len(retrieved_docs)
            
        except Exception as e:
            logger.error(f"RAG生成失败: {e}")
            self.error_count += 1
            return "", 0.0, 0.0, 0
    
    def evaluate_single_sample(self, data: Dict[str, Any], idx: int) -> Dict[str, Any]:
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
        
        # 1. 使用RAG生成答案
        predicted_answer, retrieval_time, generation_time, num_docs = self.generate_answer_with_rag(instruction)
        
        total_processing_time = time.time() - start_time
        
        # 2. 评估答案质量（与E1/E2/E3相同的评分标准）
        if true_task_type == 'biomass':
            metrics = self.metrics_calculator.calculate_biomass_metrics(
                predicted_answer=predicted_answer,
                reference=reference,
                instruction=instruction,
                equation_db=None  # E4没有方程库
            )
        else:
            metrics = self.metrics_calculator.calculate_qa_metrics(
                predicted=predicted_answer,
                reference=reference,
                question=instruction,
                task_type=true_task_type
            )
        
        # 3. 更新统计
        self.task_stats[true_task_type]['total'] += 1
        self.task_stats[true_task_type]['answer_correct'] += metrics['score']
        self.task_stats[true_task_type]['scores'].append(metrics['score'])
        self.task_stats[true_task_type]['response_lengths'].append(len(predicted_answer))
        self.task_stats[true_task_type]['processing_times'].append(total_processing_time)
        self.task_stats[true_task_type]['retrieval_times'].append(retrieval_time)
        self.task_stats[true_task_type]['generation_times'].append(generation_time)
        self.task_stats[true_task_type]['retrieved_docs_counts'].append(num_docs)
        
        # 记录详细维度
        if true_task_type == 'biomass':
            details = metrics.get('details', {})
            self.task_stats[true_task_type]['species_scores'].append(details.get('species', 0))
            self.task_stats[true_task_type]['region_scores'].append(details.get('region', 0))
            self.task_stats[true_task_type]['component_scores'].append(details.get('component', 0))
            self.task_stats[true_task_type]['variable_scores'].append(details.get('variables', 0))
            self.task_stats[true_task_type]['form_scores'].append(details.get('form', 0))
            self.task_stats[true_task_type]['coefficient_scores'].append(details.get('coefficient', 0))
        elif true_task_type in ['forestry_qa', 'general_chat']:
            self.task_stats[true_task_type]['accuracy_scores'].append(metrics.get('accuracy', 0))
            self.task_stats[true_task_type]['completeness_scores'].append(metrics.get('completeness', 0))
            self.task_stats[true_task_type]['relevance_scores'].append(metrics.get('relevance', 0))
        
        # 4. 构建结果
        result = {
            'index': idx,
            'true_task_type': true_task_type,
            'instruction': instruction,
            'reference': reference,
            'predicted_answer': predicted_answer,
            'rag_info': {
                'num_retrieved_docs': num_docs,
                'retrieval_time': retrieval_time
            },
            'metrics': metrics,
            'score': metrics['score'],
            'timing': {
                'total_time': total_processing_time,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time
            },
            'response_length': len(predicted_answer),
            'metadata': data.get('metadata', {})
        }
        
        return result
    
    def run_evaluation(self):
        """运行完整评估"""
        logger.info("=" * 80)
        logger.info("🚀 开始E4实验评估...")
        logger.info(f"  实验: 纯RAG系统（基座模型）")
        logger.info(f"  模型: Qwen基座（未微调）")
        logger.info(f"  评估方法: 与E1/E2/E3相同")
        logger.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 加载测试数据
        test_data = self.load_test_data()
        total_samples = len(test_data)
        
        # 打开文件句柄
        detailed_file_handle = open(self.detailed_file, 'w', encoding='utf-8')
        
        start_time = time.time()
        
        # 逐样本评估
        for idx, data in enumerate(test_data):
            result = self.evaluate_single_sample(data, idx)
            
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
                
                current_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0
                
                logger.info(
                    f"  进度: {idx + 1}/{total_samples} ({(idx+1)/total_samples*100:.1f}%) | "
                    f"答案准确率: {current_accuracy:.2%} | "
                    f"速度: {samples_per_sec:.2f} 样本/秒 | "
                    f"预计剩余: {eta_seconds/60:.1f} 分钟"
                )
        
        detailed_file_handle.close()
        
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
            'experiment_name': 'E4-Pure-RAG',
            'model_type': 'Qwen_Base_PureRAG',
            'evaluation_method': 'Multidimensional_LLM_Fallback',
            'total_samples': len(self.results),
            'error_count': self.error_count,
            'empty_response_count': self.empty_response_count,
            'task_performance': {}
        }
        
        # 计算每个任务的详细指标
        for task_type, stats in self.task_stats.items():
            if stats['total'] == 0:
                continue
            
            task_metrics = {
                'total_samples': stats['total'],
                'answer_accuracy': stats['answer_correct'] / stats['total'],
                'average_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                'std_score': np.std(stats['scores']) if stats['scores'] else 0.0,
                'average_response_length': np.mean(stats['response_lengths']) if stats['response_lengths'] else 0.0,
                'average_total_time': np.mean(stats['processing_times']) if stats['processing_times'] else 0.0,
                'average_retrieval_time': np.mean(stats['retrieval_times']) if stats['retrieval_times'] else 0.0,
                'average_generation_time': np.mean(stats['generation_times']) if stats['generation_times'] else 0.0,
                'average_retrieved_docs': np.mean(stats['retrieved_docs_counts']) if stats['retrieved_docs_counts'] else 0.0
            }
            
            if task_type == 'biomass':
                task_metrics.update({
                    'average_species_match': np.mean(stats['species_scores']) if stats['species_scores'] else 0.0,
                    'average_region_match': np.mean(stats['region_scores']) if stats['region_scores'] else 0.0,
                    'average_component_match': np.mean(stats['component_scores']) if stats['component_scores'] else 0.0,
                    'average_variable_match': np.mean(stats['variable_scores']) if stats['variable_scores'] else 0.0,
                    'average_form_match': np.mean(stats['form_scores']) if stats['form_scores'] else 0.0,
                    'average_coefficient_similarity': np.mean(stats['coefficient_scores']) if stats['coefficient_scores'] else 0.0
                })
            elif task_type in ['forestry_qa', 'general_chat']:
                task_metrics.update({
                    'average_accuracy': np.mean(stats['accuracy_scores']) if stats['accuracy_scores'] else 0.0,
                    'average_completeness': np.mean(stats['completeness_scores']) if stats['completeness_scores'] else 0.0,
                    'average_relevance': np.mean(stats['relevance_scores']) if stats['relevance_scores'] else 0.0
                })
            
            logger.info(f"\n📈 {task_type} 性能:")
            logger.info(f"  - 样本数: {task_metrics['total_samples']}")
            logger.info(f"  - 答案准确率: {task_metrics['answer_accuracy']:.2%}")
            logger.info(f"  - 平均得分: {task_metrics['average_score']:.2f} ± {task_metrics['std_score']:.2f}")
            logger.info(f"  - 平均检索文档数: {task_metrics['average_retrieved_docs']:.1f}")
            
            summary['task_performance'][task_type] = task_metrics
        
        # 计算整体指标
        total_samples = sum(s['total'] for s in self.task_stats.values())
        
        if total_samples > 0:
            overall_metrics = {
                'answer_accuracy': sum(s['answer_correct'] for s in self.task_stats.values()) / total_samples,
                'average_score': np.mean([score for s in self.task_stats.values() for score in s['scores']]),
                'average_total_time': np.mean([t for s in self.task_stats.values() for t in s['processing_times']]),
                'average_retrieval_time': np.mean([t for s in self.task_stats.values() for t in s['retrieval_times']]),
                'average_generation_time': np.mean([t for s in self.task_stats.values() for t in s['generation_times']])
            }
            
            summary['overall_performance'] = overall_metrics
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 整体性能:")
            logger.info(f"  - 总样本数: {total_samples}")
            logger.info(f"  - 答案准确率: {overall_metrics['answer_accuracy']:.2%}")
            logger.info(f"  - 平均得分: {overall_metrics['average_score']:.2f}")
            logger.info(f"  - 平均总耗时: {overall_metrics['average_total_time']:.3f} 秒/样本")
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """保存结果"""
        logger.info("=" * 80)
        logger.info(f"💾 保存结果到: {self.output_file}")
        
        full_results = {
            'metadata': {
                'experiment': 'E4-Pure-RAG',
                'timestamp': datetime.now().isoformat(),
                'base_model': self.base_model_path,
                'lora_model': None,
                'test_file': self.test_file,
                'evaluation_method': 'Multidimensional_LLM_Fallback',
                'note': 'Using base Qwen model (not fine-tuned) + Pure RAG'
            },
            'summary': summary,
            'sample_results': self.results[:200]
        }
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 汇总结果已保存: {self.output_file}")
            logger.info(f"✅ 详细结果已保存: {self.detailed_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}", exc_info=True)
    
    def run(self):
        """运行完整的E4实验"""
        logger.info("🎯 开始执行 E4-Pure-RAG 实验")
        logger.info(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"关键设计: 基座模型 + 纯RAG（无专门模块）")
        
        # 1. 加载所有组件
        self.load_components()
        
        # 2. 运行评估
        self.run_evaluation()
        
        # 3. 计算汇总指标
        summary = self.calculate_summary_metrics()
        
        # 4. 保存结果
        self.save_results(summary)
        
        logger.info("=" * 80)
        logger.info("🎉 E4-Pure-RAG 实验完成!")
        logger.info("下一步: 对比E1/E2/E3/E4结果")
        logger.info("=" * 80)


if __name__ == "__main__":
    experiment = E4PureRAGExperiment()
    experiment.run()