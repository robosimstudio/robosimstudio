/* SPDX-License-Identifier: Apache-2.0 */
/* A pass-through Vulkan layer that shows the application one GPU only.
 *
 * Headless Chromium (ANGLE on Vulkan) takes the first device the loader lists, GPU 0, whatever
 * CUDA_VISIBLE_DEVICES says: measured on an eight-card box, a page recorded under
 * CUDA_VISIBLE_DEVICES=1 put its 240 MB on GPU 0, and ANGLE's gl-egl backend did the same.
 * Loaded as an implicit layer through VK_ADD_IMPLICIT_LAYER_PATH, this filters
 * vkEnumeratePhysicalDevices(/Groups) down to the device whose deviceUUID equals $RSS_VK_UUID
 * (nvidia-smi's GPU-xxxxxxxx-... form or bare hex). No header dependency: the few loader
 * structs are declared here with the layout of vk_layer.h.
 *
 * scripts/recording/page.py compiles it on first use (cc -shared -fPIC -O2) into
 * outputs/record/.vkpick/ beside its manifest, and points the browser at it.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#define VKAPI __attribute__((visibility("default")))
typedef int32_t VkResult;
typedef void *VkInstance, *VkPhysicalDevice, *VkDevice;
typedef void (*PFN_void)(void);
typedef PFN_void (*PFN_GIPA)(VkInstance, const char *);
typedef PFN_void (*PFN_GDPA)(VkDevice, const char *);
typedef VkResult (*PFN_CreateInstance)(const void *, const void *, VkInstance *);
typedef VkResult (*PFN_CreateDevice)(VkPhysicalDevice, const void *, const void *, VkDevice *);
typedef VkResult (*PFN_EnumPD)(VkInstance, uint32_t *, VkPhysicalDevice *);
typedef void (*PFN_GetProps2)(VkPhysicalDevice, void *);

typedef struct { int32_t sType; const void *pNext; } Base;
typedef struct InstLink { struct InstLink *pNext; PFN_GIPA gipa; void *gpdpa; } InstLink;
typedef struct { int32_t sType; const void *pNext; int32_t function; union { InstLink *link; void *p; } u; } InstCI;
typedef struct DevLink { struct DevLink *pNext; PFN_GIPA gipa; PFN_GDPA gdpa; } DevLink;
typedef struct { int32_t sType; const void *pNext; int32_t function; union { DevLink *link; void *p; } u; } DevCI;
typedef struct { int32_t sType; void *pNext; uint32_t version; PFN_GIPA gipa; PFN_GDPA gdpa; void *gpdpa; } Negotiate;

/* VkPhysicalDeviceProperties2 with room for the core struct, chained to VkPhysicalDeviceIDProperties */
typedef struct { int32_t sType; void *pNext; uint8_t deviceUUID[16]; uint8_t driverUUID[16]; uint8_t luid[8]; uint32_t mask; uint32_t valid; } IDProps;
typedef struct { int32_t sType; void *pNext; uint8_t props[4096]; } Props2;
typedef struct { int32_t sType; void *pNext; uint32_t count; VkPhysicalDevice devs[32]; uint32_t subset; } Group;

static PFN_GIPA next_gipa = 0;
static PFN_GDPA next_gdpa = 0;

static int want_uuid(uint8_t out[16]) {
    const char *s = getenv("RSS_VK_UUID");
    if (!s) return 0;
    if (!strncmp(s, "GPU-", 4)) s += 4;
    int n = 0;
    for (; *s && n < 32; s++) {
        if (*s == '-') continue;
        if (!isxdigit((unsigned char)*s)) return 0;
        int v = isdigit((unsigned char)*s) ? *s - '0' : (tolower((unsigned char)*s) - 'a' + 10);
        if (n % 2 == 0) out[n / 2] = (uint8_t)(v << 4); else out[n / 2] |= (uint8_t)v;
        n++;
    }
    return n == 32;
}

static int is_wanted(VkInstance inst, VkPhysicalDevice pd, const uint8_t uuid[16]) {
    PFN_GetProps2 get = (PFN_GetProps2)next_gipa(inst, "vkGetPhysicalDeviceProperties2");
    if (!get) get = (PFN_GetProps2)next_gipa(inst, "vkGetPhysicalDeviceProperties2KHR");
    if (!get) return -1;
    IDProps id; memset(&id, 0, sizeof id); id.sType = 1000071004;
    Props2 p; memset(&p, 0, sizeof p); p.sType = 1000059001; p.pNext = &id;
    get(pd, &p);
    return memcmp(id.deviceUUID, uuid, 16) == 0;
}

static VkResult my_EnumeratePhysicalDevices(VkInstance inst, uint32_t *count, VkPhysicalDevice *out) {
    PFN_EnumPD next = (PFN_EnumPD)next_gipa(inst, "vkEnumeratePhysicalDevices");
    uint8_t uuid[16];
    if (!want_uuid(uuid)) return next(inst, count, out);
    uint32_t n = 0;
    VkResult r = next(inst, &n, 0);
    if (r != 0 || n == 0) return next(inst, count, out);
    VkPhysicalDevice *all = calloc(n, sizeof *all);
    r = next(inst, &n, all);
    if (r < 0) { free(all); return r; }
    VkPhysicalDevice keep[64]; uint32_t k = 0;
    for (uint32_t i = 0; i < n && k < 64; i++) {
        int w = is_wanted(inst, all[i], uuid);
        if (w < 0) { free(all); return next(inst, count, out); }
        if (w) keep[k++] = all[i];
    }
    free(all);
    if (k == 0) { fprintf(stderr, "[pickgpu] no device matches RSS_VK_UUID; passing all through\n"); return next(inst, count, out); }
    if (!out) { *count = k; return 0; }
    uint32_t m = *count < k ? *count : k;
    memcpy(out, keep, m * sizeof *out);
    *count = m;
    return m < k ? 5 /* VK_INCOMPLETE */ : 0;
}

static VkResult my_EnumeratePhysicalDeviceGroups(VkInstance inst, uint32_t *count, Group *out) {
    typedef VkResult (*PFN_G)(VkInstance, uint32_t *, Group *);
    PFN_G next = (PFN_G)next_gipa(inst, "vkEnumeratePhysicalDeviceGroups");
    if (!next) next = (PFN_G)next_gipa(inst, "vkEnumeratePhysicalDeviceGroupsKHR");
    uint8_t uuid[16];
    if (!want_uuid(uuid)) return next(inst, count, out);
    uint32_t n = 0;
    VkResult r = next(inst, &n, 0);
    if (r != 0 || n == 0) return next(inst, count, out);
    Group *all = calloc(n, sizeof *all);
    for (uint32_t i = 0; i < n; i++) all[i].sType = 1000070000;
    r = next(inst, &n, all);
    if (r < 0) { free(all); return r; }
    Group keep[64]; uint32_t k = 0;
    for (uint32_t i = 0; i < n && k < 64; i++) {
        int hit = 0;
        for (uint32_t j = 0; j < all[i].count; j++) {
            int w = is_wanted(inst, all[i].devs[j], uuid);
            if (w < 0) { free(all); return next(inst, count, out); }
            hit |= w;
        }
        if (hit) keep[k++] = all[i];
    }
    free(all);
    if (k == 0) return next(inst, count, out);
    if (!out) { *count = k; return 0; }
    uint32_t m = *count < k ? *count : k;
    for (uint32_t i = 0; i < m; i++) {
        void *pn = out[i].pNext;
        out[i] = keep[i];
        out[i].pNext = pn;
    }
    *count = m;
    return m < k ? 5 : 0;
}

static VkResult my_CreateInstance(const void *ci, const void *alloc, VkInstance *inst) {
    InstCI *link = (InstCI *)((const Base *)ci)->pNext;
    while (link && !(link->sType == 47 && link->function == 0)) link = (InstCI *)link->pNext;
    if (!link) return -3; /* VK_ERROR_INITIALIZATION_FAILED */
    PFN_GIPA gipa = link->u.link->gipa;
    link->u.link = link->u.link->pNext;
    PFN_CreateInstance create = (PFN_CreateInstance)gipa(0, "vkCreateInstance");
    if (!create) return -3;
    VkResult r = create(ci, alloc, inst);
    if (r == 0) next_gipa = gipa;
    return r;
}

static VkResult my_CreateDevice(VkPhysicalDevice pd, const void *ci, const void *alloc, VkDevice *dev) {
    DevCI *link = (DevCI *)((const Base *)ci)->pNext;
    while (link && !(link->sType == 48 && link->function == 0)) link = (DevCI *)link->pNext;
    if (!link) return -3;
    PFN_GIPA gipa = link->u.link->gipa;
    PFN_GDPA gdpa = link->u.link->gdpa;
    link->u.link = link->u.link->pNext;
    PFN_CreateDevice create = (PFN_CreateDevice)gipa(0, "vkCreateDevice");
    if (!create) return -3;
    VkResult r = create(pd, ci, alloc, dev);
    if (r == 0) next_gdpa = gdpa;
    return r;
}

VKAPI PFN_void pick_GetDeviceProcAddr(VkDevice dev, const char *name);
VKAPI PFN_void pick_GetInstanceProcAddr(VkInstance inst, const char *name) {
    if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_void)pick_GetInstanceProcAddr;
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_void)pick_GetDeviceProcAddr;
    if (!strcmp(name, "vkCreateInstance")) return (PFN_void)my_CreateInstance;
    if (!strcmp(name, "vkCreateDevice")) return (PFN_void)my_CreateDevice;
    if (!strcmp(name, "vkEnumeratePhysicalDevices")) return (PFN_void)my_EnumeratePhysicalDevices;
    if (!strcmp(name, "vkEnumeratePhysicalDeviceGroups") || !strcmp(name, "vkEnumeratePhysicalDeviceGroupsKHR"))
        return (PFN_void)my_EnumeratePhysicalDeviceGroups;
    return next_gipa ? next_gipa(inst, name) : 0;
}

VKAPI PFN_void pick_GetDeviceProcAddr(VkDevice dev, const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_void)pick_GetDeviceProcAddr;
    return next_gdpa ? next_gdpa(dev, name) : 0;
}

VKAPI VkResult vkNegotiateLoaderLayerInterfaceVersion(Negotiate *n) {
    if (n->version > 2) n->version = 2;
    n->gipa = pick_GetInstanceProcAddr;
    n->gdpa = pick_GetDeviceProcAddr;
    n->gpdpa = 0;
    return 0;
}
