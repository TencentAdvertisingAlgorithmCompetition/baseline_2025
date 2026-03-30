#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/MetricType.h>
#include <gflags/gflags.h>
#include <optional>


static const char* kFaissIndexType_HNSW = "hnsw";
static const char* kFaissIndexType_BruteForce = "brute-force";

DEFINE_string(dataset_vector_file_path, "resources/embedding.fbin", "数据集向量文件路径");
DEFINE_string(dataset_id_file_path, "resources/id.u64bin", "数据集id文件路径");
DEFINE_string(faiss_index_type, kFaissIndexType_HNSW, "faiss索引类型, 默认hnsw, 可选brute-force");
DEFINE_int32(faiss_metric_type, 1, "faiss距离度量类型, 1-L2");
DEFINE_int32(faiss_M, 16, "每个节点在构建时保留的最近邻数量");
DEFINE_int32(faiss_ef_construction, 80, "构建时搜索深度");
DEFINE_int32(query_ef_search, 32, "检索时搜索深度");
DEFINE_int32(query_ann_top_k, 10, "ann-topk");
DEFINE_string(query_vector_file_path, "resources/query.fbin", "query向量文件路径");
DEFINE_string(query_one_vector, "", "逗号分割的明文float数组");
DEFINE_string(result_id_file_path, "result/id.u64bin", "结果id文件路径");

/**
 * 解析二进制文件数据的模板函数
 *
 * @param filename 文件路径
 * @param data 输出数据
 * @return true 解析成功, false 解析失败
 */
template <typename T>
bool ParseBinaryFile(const std::string& filename, std::vector<T>& data, uint32_t& num_points,
                     uint32_t& num_dimensions) {
  // 打开二进制文件
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return false;
  }

  // 获取文件大小
  const size_t file_size = file.tellg();
  file.seekg(0);

  if (file_size < 8) {
    std::cerr << "文件太小，无法读取头部信息: " << filename << std::endl;
    return false;
  }

  // 读取头部信息
  file.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
  file.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));

  // 打印头部信息
  std::cout << "num_points=" << num_points << ", num_dimensions=" << num_dimensions << std::endl;

  if (num_points == 0) {
    std::cerr << "无效的向量数量: 0" << std::endl;
    return false;
  }
  if (num_dimensions == 0) {
    std::cerr << "无效的向量维度: 0" << std::endl;
    return false;
  }

  // 计算预期文件大小
  const size_t expected_size = 8 + num_points * num_dimensions * sizeof(T);

  if (file_size != expected_size) {
    std::cerr << "文件大小与内容描述不符: 预期=" << expected_size << ", 实际=" << file_size << std::endl;
    return false;
  }

  // 根据数据类型进行解析
  if constexpr (std::is_same_v<T, faiss::idx_t>) {
    // 处理 uint32_t 数组情况
    if (num_dimensions != 1) {
      std::cerr << "文件指定维度为 " << num_dimensions << "，但预期 uint32_t 向量的维度应为 1" << std::endl;
      return false;
    }

    data.resize(num_points);
    file.read(reinterpret_cast<char*>(data.data()), num_points * sizeof(T));

    return file.good();
  } else if constexpr (std::is_same_v<T, float>) {
    // 处理浮点数向量情况
    if (num_dimensions <= 1) {
      std::cerr << "文件指定维度为 " << num_dimensions << "，但预期浮点向量的维度应大于 1" << std::endl;
      return false;
    }

    data.resize(num_points * num_dimensions);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));

    return file.good();
  } else {
    static_assert(std::is_same_v<T, faiss::idx_t> || std::is_same_v<T, float>,
                  "不支持的模板类型: 仅支持 std::vector<faiss::idx_t> 或 "
                  "std::vector<std::vector<float>>");
    return false;
  }
}

bool IsMetricTypeValid(int32_t value) {
  static const std::unordered_set<int> validValues = {faiss::METRIC_INNER_PRODUCT,
                                                      faiss::METRIC_L2,
                                                      faiss::METRIC_L1,
                                                      faiss::METRIC_Linf,
                                                      faiss::METRIC_Lp,
                                                      faiss::METRIC_Canberra,
                                                      faiss::METRIC_BrayCurtis,
                                                      faiss::METRIC_JensenShannon,
                                                      faiss::METRIC_Jaccard,
                                                      faiss::METRIC_NaNEuclidean,
                                                      faiss::METRIC_ABS_INNER_PRODUCT};
  return validValues.count(value) > 0;
}

// 打印索引信息
void print_index_info(const faiss::Index* index) {
  if (index) {
    std::cout << "Faiss Index Information:\n";
    std::cout << "  - Dimensions: " << index->d << "\n";
    std::cout << "  - Training Size: " << index->ntotal << "\n";
    std::cout << "  - Metric Type: " << index->metric_type << "\n";
  } else {
    std::cerr << "Error: Not an Faiss index\n";
  }
}

std::optional<std::vector<float>> ParseCommaSeparatedFloats(const std::string& input) {
  std::vector<float> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, ',')) {
    try {
      result.push_back(std::stof(token));
    } catch (const std::exception& e) {
      std::cerr << "Warning: Failed to parse float from token: '" << token << "'\n";
      return std::nullopt;
    }
  }

  return result;
}

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  // 获取所有标志的键值对
  std::vector<GFLAGS_NAMESPACE::CommandLineFlagInfo> all_flags;
  GFLAGS_NAMESPACE::GetAllFlags(&all_flags);
  // 遍历并打印所有参数
  for (const auto& pair : all_flags) {
    std::cout << pair.name << "=" << pair.current_value << std::endl;
  }

  if (!IsMetricTypeValid(FLAGS_faiss_metric_type)) {
    std::cerr << "faiss_metric_type=" << FLAGS_faiss_metric_type << " 不是有效的值" << std::endl;
    return 1;
  }

  uint32_t num_points_idx = 0;
  uint32_t num_dimensions_idx = 0;
  std::vector<faiss::idx_t> id_data;
  if (!ParseBinaryFile(FLAGS_dataset_id_file_path, id_data, num_points_idx, num_dimensions_idx)) {
    std::cerr << "id file:" << FLAGS_dataset_id_file_path << " 打开失败" << std::endl;
    return 2;
  }
  // 打印前10个
  std::cout << "id file:" << std::endl;
  for (auto i = 0; i < std::min(num_points_idx, static_cast<uint32_t>(10)); ++i) {
    std::cout << id_data[i] << std::endl;
  }
  std::cout << std::endl;

  // 示例2: 解析维度>1的文件
  uint32_t num_points_index = 0;
  uint32_t num_dimensions_index = 0;
  std::vector<float> float_data;
  if (!ParseBinaryFile(FLAGS_dataset_vector_file_path, float_data, num_points_index, num_dimensions_index)) {
    return 3;
  }
  // 打印前10个
  std::cout << "vector file:" << std::endl;
  for (auto i = 0; i < std::min(num_points_index, static_cast<uint32_t>(1)); ++i) {
    for (auto j = 1; j < num_dimensions_index; ++j) {
      std::cout << float_data[i * num_dimensions_index + j] << ", ";
    }
    std::cout << std::endl;
  }

  if (num_points_index != num_points_idx) {
    std::cerr << "数量不匹配, 向量个数: " << num_points_index << ", id个数: " << num_points_idx << std::endl;
    return 4;
  }

  // query文件
  uint32_t num_points_query = 0;
  uint32_t num_dimensions_query = 0;
  std::vector<float> query_data;

  if (!FLAGS_query_one_vector.empty()) {
    // 将FLAGS_query_one_vector按逗号分割解析为一个std::vector
    auto query_one_vector = ParseCommaSeparatedFloats(FLAGS_query_one_vector);
    if (!query_one_vector) {
      std::cerr << "query_one_vector=" << FLAGS_query_one_vector << " 不是有效的值" << std::endl;
      return 5;
    }
    num_points_query = 1;
    num_dimensions_query = query_one_vector->size();
    query_data.resize(num_dimensions_query * num_points_query);
    std::copy(query_one_vector->begin(), query_one_vector->end(), query_data.begin());
    // 打印query_one_vector
    std::cout << "query_one_vector:" << std::endl;
    for (auto i = 0; i < 1; ++i) {
      for (auto j = 1; j < num_dimensions_query; ++j) {
        std::cout << query_data[i * num_dimensions_query + j] << ", ";
      }
      std::cout << std::endl;
    }
  } else {
    if (!ParseBinaryFile(FLAGS_query_vector_file_path, query_data, num_points_query, num_dimensions_query)) {
      return 5;
    }
    // 打印前10个
    std::cout << "query file:" << std::endl;
    for (auto i = 0; i < std::min(num_points_query, static_cast<uint32_t>(1)); ++i) {
      for (auto j = 1; j < num_dimensions_query; ++j) {
        std::cout << query_data[i * num_dimensions_query + j] << ", ";
      }
      std::cout << std::endl;
    }
  }
  if (num_dimensions_query != num_dimensions_index) {
    std::cerr << "维度不匹配, 数据集向量维度: " << num_dimensions_index << ", query维度: " << num_dimensions_query
              << std::endl;
    return 6;
  }

  faiss::Index* index = nullptr;
  if (FLAGS_faiss_index_type == kFaissIndexType_HNSW) {
    faiss::IndexHNSWFlat* hnsw_index = new faiss::IndexHNSWFlat(
        num_dimensions_index, FLAGS_faiss_M, static_cast<faiss::MetricType>(FLAGS_faiss_metric_type));
    hnsw_index->hnsw.efConstruction = FLAGS_faiss_ef_construction;  // 构建时搜索深度
    hnsw_index->hnsw.efSearch = FLAGS_query_ef_search;              // 查询时搜索深度
    index = hnsw_index;
    std::cout << "构建HNSW索引" << std::endl;
  } else if (FLAGS_faiss_index_type == kFaissIndexType_BruteForce) {
    faiss::IndexFlat* brute_force_index =
        new faiss::IndexFlat(num_dimensions_index, static_cast<faiss::MetricType>(FLAGS_faiss_metric_type));
    index = brute_force_index;
    std::cout << "构建BruteForce索引" << std::endl;
  } else {
    std::cerr << "faiss_index_type=" << FLAGS_faiss_index_type << " 类型暂不支持" << std::endl;
    return 7;
  }
  // 添加数据到索引 (索引会自动训练)
  faiss::IndexIDMap index_wrapper(index);
  auto begin = std::chrono::steady_clock::now();
  index_wrapper.add_with_ids(num_points_index, float_data.data(), id_data.data());
  auto end = std::chrono::steady_clock::now();
  std::cout << "构建索引耗时: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s"
            << std::endl;
  print_index_info(index_wrapper.index);

  // 准备搜索结果
  std::vector<faiss::idx_t> labels(FLAGS_query_ann_top_k * num_points_query);
  std::vector<float> distances(FLAGS_query_ann_top_k * num_points_query);

  // 执行搜索
  auto start_search = std::chrono::steady_clock::now();
  index_wrapper.search(num_points_query, query_data.data(), FLAGS_query_ann_top_k, distances.data(), labels.data());
  auto end_search = std::chrono::steady_clock::now();
  std::cout << "检索" << num_points_query << "个查询, 每个查询" << FLAGS_query_ann_top_k << "个结果, 总耗时: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_search - start_search).count() << " ms"
            << std::endl;

  // 结果写文件
  if (!std::filesystem::exists("./result")) {
    std::filesystem::create_directory("./result");
  }

  std::vector<faiss::idx_t> result_ids(FLAGS_query_ann_top_k * num_points_query);
  for (auto i = 0; i < num_points_query; ++i) {
    for (auto j = 0; j < FLAGS_query_ann_top_k; ++j) {
      result_ids[i * FLAGS_query_ann_top_k + j] = id_data[labels[i * FLAGS_query_ann_top_k + j]];
    }
  }

  std::ofstream outfile(FLAGS_result_id_file_path, std::ios::binary);
  if (!outfile) {
      std::cerr << "Error opening file for writing: " << FLAGS_result_id_file_path << std::endl;
      return 7;
  }
  // Write result
  outfile.write(reinterpret_cast<const char*>(&num_points_query), sizeof(num_points_query));
  outfile.write(reinterpret_cast<const char*>(&FLAGS_query_ann_top_k), sizeof(FLAGS_query_ann_top_k));
  outfile.write(reinterpret_cast<const char*>(result_ids.data()), result_ids.size() * sizeof(result_ids[0]));
  // Check if write was successful
  if (!outfile) {
      std::cerr << "Error writing to file: " << FLAGS_result_id_file_path << std::endl;
  }  
  outfile.close();
  return 0;
}
