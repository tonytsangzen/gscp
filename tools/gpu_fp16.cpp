// gpu_fp16.cpp — real GPU capability probe: fp16 tiled GEMM via Vulkan compute.
// Measures the TRUE GFLOPS of the Mali GPU on this device (fp32 naive shader in
// gpu_probe.cpp undercounts badly; TFLite GLES delegate is emulated/slow).
// Diagnostics only — reports per-size ms + GFLOPS via logcat and as JSON.
#define LOG_TAG "gpufp16"

#include <android/log.h>
#include <jni.h>
#include <vulkan/vulkan.h>
#include "gemm16_spv.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace {

struct Ctx16 {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phy = VK_NULL_HANDLE;
    uint32_t qf = 0;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout ppl = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule sm = VK_NULL_HANDLE;
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkBuffer bA = VK_NULL_HANDLE, bB = VK_NULL_HANDLE, bC = VK_NULL_HANDLE;
    VkDeviceMemory mA = VK_NULL_HANDLE, mB = VK_NULL_HANDLE, mC = VK_NULL_HANDLE;
    void* pA = nullptr; void* pB = nullptr;
};

static bool initVk(Ctx16& c, std::string& name) {
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "fp16gemm";
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, nullptr, &c.inst) != VK_SUCCESS) return false;
    uint32_t nd = 0; vkEnumeratePhysicalDevices(c.inst, &nd, nullptr);
    if (!nd) return false;
    std::vector<VkPhysicalDevice> pds(nd);
    vkEnumeratePhysicalDevices(c.inst, &nd, pds.data()); c.phy = pds[0];
    VkPhysicalDeviceProperties prop; vkGetPhysicalDeviceProperties(c.phy, &prop);
    name = prop.deviceName ? prop.deviceName : "?";
    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(c.phy, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phy, &nqf, qfs.data());
    c.qf = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++)
        if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { c.qf = i; break; }
    if (c.qf == UINT32_MAX)
        for (uint32_t i = 0; i < nqf; i++) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qf = i; break; }
    if (c.qf == UINT32_MAX) return false;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqi{};
    dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqi.queueFamilyIndex = c.qf; dqi.queueCount = 1; dqi.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dqi;
    if (vkCreateDevice(c.phy, &dci, nullptr, &c.dev) != VK_SUCCESS) return false;
    vkGetDeviceQueue(c.dev, c.qf, 0, &c.queue);

    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sizeof(uint32_t) * kGEMM16SpvLen; smi.pCode = kGEMM16Spv;
    if (vkCreateShaderModule(c.dev, &smi, nullptr, &c.sm) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding bnd[3]{};
    for (int i = 0; i < 3; i++) { bnd[i].binding = (uint32_t)i; bnd[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bnd[i].descriptorCount = 1; bnd[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 3; dlci.pBindings = bnd;
    if (vkCreateDescriptorSetLayout(c.dev, &dlci, nullptr, &c.dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset = 0; pcr.size = 3 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &c.dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(c.dev, &plci, nullptr, &c.ppl) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = c.sm; cpi.stage.pName = "main";
    cpi.layout = c.ppl;
    if (vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &c.pipeline) != VK_SUCCESS) return false;

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci.queueFamilyIndex = c.qf;
    if (vkCreateCommandPool(c.dev, &cpci, nullptr, &c.cpool) != VK_SUCCESS) return false;
    VkDescriptorPoolSize dps{};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; dps.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    if (vkCreateDescriptorPool(c.dev, &dpci, nullptr, &c.dpool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = c.dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &c.dsl;
    if (vkAllocateDescriptorSets(c.dev, &dsai, &c.dset) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = c.cpool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(c.dev, &cai, &c.cb) != VK_SUCCESS) return false;
    return true;
}

static inline uint16_t f2h(float f) {
    union { uint32_t i; float f; } u; u.f = f;
    uint32_t x = u.i;
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exp = (x >> 23) & 0xff, mant = x & 0x7fffff;
    if (exp == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    int e = (int)exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000; uint32_t s = -e + 1; mant >>= s;
        return (uint16_t)(sign | (((mant + 0xfff + ((mant >> 13) & 1)) >> 13)));
    }
    if (e > 0) { mant >>= 13; if (mant == 0x400) { mant = 0; e++; } }
    mant &= 0x3ff;
    return (uint16_t)(sign | (e << 10) | mant);
}

static bool setupBuffers(Ctx16& c, size_t M, size_t N, size_t K) {
    VkDeviceSize aB = M * K * 2, bB = K * N * 2, cB = M * N * 2;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto mk = [&](VkBuffer& b, VkDeviceMemory& m, void** p, VkDeviceSize sz) -> bool {
        bi.size = sz;
        if (vkCreateBuffer(c.dev, &bi, nullptr, &b) != VK_SUCCESS) return false;
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(c.dev, b, &req);
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(c.phy, &mp);
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }
        if (mt == UINT32_MAX) for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { mt = i; break; }
        if (mt == UINT32_MAX) return false;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = req.size; mai.memoryTypeIndex = mt;
        if (vkAllocateMemory(c.dev, &mai, nullptr, &m) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(c.dev, b, m, 0) != VK_SUCCESS) return false;
        return vkMapMemory(c.dev, m, 0, VK_WHOLE_SIZE, 0, p) == VK_SUCCESS;
    };
    if (!mk(c.bA, c.mA, &c.pA, aB)) return false;
    if (!mk(c.bB, c.mB, &c.pB, bB)) return false;
    if (!mk(c.bC, c.mC, &c.pA, cB)) return false;  // C ptr unused
    std::vector<uint16_t> A(M * K), B(K * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = f2h(0.0001f * (float)((i * 131) % 1000));
    for (size_t i = 0; i < B.size(); i++) B[i] = f2h(0.0001f * (float)((i * 263) % 1000));
    memcpy(c.pA, A.data(), M * K * 2);
    memcpy(c.pB, B.data(), K * N * 2);
    return true;
}

static void updateDesc(Ctx16& c, size_t M, size_t N, size_t K) {
    VkDescriptorBufferInfo info[3] = {{c.bA,0,M*K*2},{c.bB,0,K*N*2},{c.bC,0,M*N*2}};
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) { w[i] = {}; w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = c.dset;
        w[i].dstBinding = (uint32_t)i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &info[i]; }
    vkUpdateDescriptorSets(c.dev, 3, w, 0, nullptr);
}

static void recordCb16(Ctx16& c, size_t M, size_t N, size_t K) {
    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(c.cb, &bi);
    vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeline);
    vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.ppl, 0, 1, &c.dset, 0, nullptr);
    uint32_t mnmknk[3] = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
    vkCmdPushConstants(c.cb, c.ppl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mnmknk), mnmknk);
    vkCmdDispatch(c.cb, (uint32_t)((M + 15) / 16), (uint32_t)((N + 15) / 16), 1);
    vkEndCommandBuffer(c.cb);
}

static void destroy(Ctx16& c) {
    if (c.mA) { vkUnmapMemory(c.dev, c.mA); vkFreeMemory(c.dev, c.mA, nullptr); }
    if (c.mB) vkFreeMemory(c.dev, c.mB, nullptr);
    if (c.mC) vkFreeMemory(c.dev, c.mC, nullptr);
    if (c.bA) vkDestroyBuffer(c.dev, c.bA, nullptr);
    if (c.bB) vkDestroyBuffer(c.dev, c.bB, nullptr);
    if (c.bC) vkDestroyBuffer(c.dev, c.bC, nullptr);
    if (c.cb) vkFreeCommandBuffers(c.dev, c.cpool, 1, &c.cb);
    if (c.cpool) vkDestroyCommandPool(c.dev, c.cpool, nullptr);
    if (c.dpool) vkDestroyDescriptorPool(c.dev, c.dpool, nullptr);
    if (c.dsl) vkDestroyDescriptorSetLayout(c.dev, c.dsl, nullptr);
    if (c.ppl) vkDestroyPipelineLayout(c.dev, c.ppl, nullptr);
    if (c.pipeline) vkDestroyPipeline(c.dev, c.pipeline, nullptr);
    if (c.sm) vkDestroyShaderModule(c.dev, c.sm, nullptr);
    if (c.dev) vkDestroyDevice(c.dev, nullptr);
    if (c.inst) vkDestroyInstance(c.inst, nullptr);
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_nnapibench_NativeBridge_nativeFp16Test(JNIEnv* env, jobject) {
    Ctx16 c;
    std::string name;
    if (!initVk(c, name)) return env->NewStringUTF("{\"state\":\"init_failed\"}");
    LOGI("fp16 GEMM on %s", name.c_str());
    std::string j = "{\"state\":\"ok\",\"device\":\"" + name + "\",\"bench\":[";
    bool any = false;
    size_t szs[3][3] = {{1024,1024,1024},{512,512,512},{2048,2048,2048}};
    for (auto& s : szs) {
        size_t M=s[0], N=s[1], K=s[2];
        if (!setupBuffers(c, M, N, K)) { LOGW("fp16 setup fail %zu", M); continue; }
        updateDesc(c, M, N, K); recordCb16(c, M, N, K);
        VkSubmitInfo si{}; si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount=1; si.pCommandBuffers=&c.cb;
        for (int w=0;w<4;w++){ vkQueueSubmit(c.queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(c.queue); }
        int iters=0; double total=0, mn=1e18;
        auto t0=std::chrono::steady_clock::now();
        while (iters<20000) {
            vkQueueSubmit(c.queue,1,&si,VK_NULL_HANDLE);
            auto a=std::chrono::steady_clock::now(); vkQueueWaitIdle(c.queue);
            auto b=std::chrono::steady_clock::now();
            double dt=std::chrono::duration<double>(b-a).count(); total+=dt; if(dt<mn)mn=dt; iters++;
            if (iters>=5 && std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>=1.0) break;
        }
        destroy(c);  // frees A/B/C; next setup allocates
        if (iters>=5) {
            double avgMs=(total/iters)*1000.0, minMs=mn*1000.0;
            double gflops=(2.0*M*N*K/1e9)/(total/iters);
            LOGI("fp16 GEMM(%zu^3): avg=%.2fms min=%.2fms GFLOPS=%.1f", M, avgMs, minMs, gflops);
            if (any) j += ","; any = true;
            j += "{\"name\":\"fp16GEMM(" + std::to_string(M) + "^3)\",\"avgMs\":" + std::to_string(avgMs)
               + ",\"minMs\":" + std::to_string(minMs) + ",\"gflops\":" + std::to_string(gflops) + "}";
        }
    }
    destroy(c);
    j += "]}";
    return env->NewStringUTF(j.c_str());
}