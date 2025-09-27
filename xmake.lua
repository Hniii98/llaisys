add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

-- add_includedirs("include")
add_includedirs("include", "src", {public = true})

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    if is_plat("windows") then
        add_includedirs("$(env CUDA_PATH)/include", {public = true})
        --add_includedirs("$(env CUDA_PATH)/include/cccl", {public = true})
        --add_linkdirs("$(env CUDA_PATH)/lib/x64")

        add_cuflags("-Xcompiler=/utf-8,/MD", {force = true})


    else
        add_includedirs("/usr/local/cuda/include", {public = true})
        add_linkdirs("/usr/local/cuda/lib64")
    end

    
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end




    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia", {public = true})
    end
    

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia", {public = true})
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

-- 模型集合（目前只有 qwen2）
target("llaisys-models")
    set_kind("static")
    add_deps("llaisys-tensor", "llaisys-ops")  -- 模型依赖 tensor/ops

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    -- 模型实现：C++ 后端
    add_files("src/models/qwen2/*.cpp")


    on_install(function (target) end)
target_end()



target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    add_deps("llaisys-models")

    

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    -- 胶水：C API 实现
    add_files("src/llaisys/models/*.cc")
    set_installdir(".")


    -- link CUDA share runtime
    if has_config("nv-gpu") then
        add_defines("ENABLE_NVIDIA_API")
        add_links("cublasLt", "cublas")
        add_syslinks("cudart")

        if is_plat("windows") then
            local cuda_path  = os.getenv("CUDA_PATH")
            --local cudnn_path = os.getenv("CUDNN_PATH")

            if cuda_path then
                add_includedirs(path.join(cuda_path, "include"), {public = true})
                add_linkdirs(path.join(cuda_path, "lib/x64"))
            end
            if cudnn_path then
                add_includedirs(path.join(cudnn_path, "include"), {public = true})
                add_linkdirs(path.join(cudnn_path, "lib/13.0/x64"))
            end
        elseif is_plat("linux") then
            -- linux

            add_includedirs("/usr/local/cuda/include", {public = true})
            add_linkdirs("/usr/local/cuda/lib64")
        end
    end

    set_policy("build.cuda.devlink", true)

    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    

        if has_config("nv-gpu") then
            -- link cuda lib
            if is_plat("windows") then
                local cuda_path  = os.getenv("CUDA_PATH")
                if cuda_path then
                    local cudabin = path.join(cuda_path, "bin/x64")
                    --local cudnnbin = path.join(cudnn_path, "bin/13.0")
                    os.trycp(path.join(cudabin, "cublas64_13.dll"),   "python/llaisys/libllaisys/")
                    os.trycp(path.join(cudabin, "cublasLt64_13.dll"), "python/llaisys/libllaisys/")
                    --os.trycp(path.join(cudnnbin, "cudnn64_9.dll"),   "python/llaisys/libllaisys/")
                else
                    print("nv-gpu enabled but CUDA_PATH is not set or CUDA tool kit is not equal to  13.0.")
                end
            elseif is_plat("linux") then
                -- linux
            end
        end
    end)
target_end()