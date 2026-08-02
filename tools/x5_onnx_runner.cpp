#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model.onnx>\n", argv[0]);
        return 1;
    }
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "x5_runner");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(2);
        so.SetInterOpNumThreads(2);
        Ort::Session session(env, argv[1], so);

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t in_count = session.GetInputCount();
        const size_t out_count = session.GetOutputCount();
        printf("inputs=%zu outputs=%zu\n", in_count, out_count);

        for (size_t i = 0; i < in_count; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            printf("input[%zu] name=%s shape=", i, name.get());
            for (auto s : shape) printf("%lld ", (long long)s);
            printf("\n");
        }

        if (in_count >= 1 && out_count >= 1) {
            auto in_type_info = session.GetInputTypeInfo(0);
            auto in_tensor_info = in_type_info.GetTensorTypeAndShapeInfo();
            auto shape = in_tensor_info.GetShape();
            int64_t total = 1;
            for (auto s : shape) {
                if (s <= 0) s = 1;
                total *= s;
            }
            std::vector<float> input_data(total, 0.0f);
            for (int64_t i = 0; i < total; ++i) input_data[i] = std::sin(i * 0.1f);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                allocator.GetInfo(), input_data.data(), input_data.size(), shape.data(), shape.size());

            auto in_name = session.GetInputNameAllocated(0, allocator);
            auto out_name = session.GetOutputNameAllocated(0, allocator);
            const char* input_names[] = { in_name.get() };
            const char* output_names[] = { out_name.get() };
            auto output = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

            auto out_info = output[0].GetTensorTypeAndShapeInfo();
            auto out_shape = out_info.GetShape();
            printf("output shape=");
            for (auto s : out_shape) printf("%lld ", (long long)s);
            printf("\n");

            const float* out_data = output[0].GetTensorData<float>();
            int64_t out_total = 1;
            for (auto s : out_shape) {
                if (s <= 0) s = 1;
                out_total *= s;
            }
            int print_n = out_total < 8 ? (int)out_total : 8;
            printf("output[0..%d]=", print_n - 1);
            for (int i = 0; i < print_n; ++i) printf("%.4f ", out_data[i]);
            printf("\n");
        }
        printf("OK\n");
        return 0;
    } catch (const std::exception& e) {
        printf("ERROR: %s\n", e.what());
        return 1;
    }
}
