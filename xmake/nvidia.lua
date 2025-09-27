target("llaisys-device-nvidia")
	set_kind("static")
	set_languages("cxx17")
	set_warnings("all", "error")
	if not is_plat("windows") then
		add_cxflags("-fPIC", "-Wno-unknown-pragmas")
	end

	add_files("../src/device/nvidia/*.cu")

	set_policy("build.cuda.devlink", true)


	on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
	set_kind("static")
	add_deps("llaisys-tensor")
	set_languages("cxx17")
	set_warnings("all", "error")
	if not is_plat("windows") then
		add_cxflags("-fPIC", "-Wno-unknown-pragmas")
	end

	add_files("../src/ops/*/nvidia/**.cu")

	add_cugencodes("native")
	
	set_policy("build.cuda.devlink", true)


	
	on_install(function (target) end)
target_end()
	