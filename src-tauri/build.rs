fn main() {
    tauri_build::build();

    // Windows: Embed Common Controls v6 manifest for test binaries
    //
    // When running `cargo test`, the generated test executables don't include
    // the standard Tauri application manifest. Without Common Controls v6,
    // `tauri::test` calls fail with STATUS_ENTRYPOINT_NOT_FOUND.
    //
    // This workaround:
    // 1. Embeds the manifest into test binaries via /MANIFEST:EMBED
    // 2. Uses /MANIFEST:NO for the main binary to avoid duplicate resources
    //    (Tauri already handles manifest embedding for the app binary)
    // These are MSVC linker options. GNU ld treats them as input files and
    // fails the build; Tauri/tauri-winres still embeds the app manifest there.
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        let manifest_path = std::path::PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"),
        )
        .join("common-controls.manifest");
        let manifest_arg = format!("/MANIFESTINPUT:{}", manifest_path.display());

        println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg={}", manifest_arg);
        // Avoid duplicate manifest resources in binary builds.
        println!("cargo:rustc-link-arg-bins=/MANIFEST:NO");
        println!("cargo:rerun-if-changed={}", manifest_path.display());
    }

    // MinGW's ld does not understand MSVC `/MANIFEST` switches. Compile the
    // same manifest with windres and attach it only to test executables; the
    // application binary gets its manifest from tauri-build.
    #[cfg(all(target_os = "windows", target_env = "gnu"))]
    {
        let manifest_path = std::path::PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"),
        )
        .join("common-controls.manifest");
        let out_dir =
            std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("missing OUT_DIR"));
        let rc_path = out_dir.join("common-controls.rc");
        let object_path = out_dir.join("common-controls.o");
        let manifest_literal = manifest_path
            .display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        std::fs::write(&rc_path, format!("1 24 \"{manifest_literal}\"\n"))
            .expect("write common-controls.rc");

        let windres = std::env::var_os("WINDRES").unwrap_or_else(|| "windres".into());
        let output = std::process::Command::new(windres)
            .arg("-O")
            .arg("coff")
            .arg("-o")
            .arg(&object_path)
            .arg(&rc_path)
            .output()
            .expect("run windres");
        assert!(
            output.status.success(),
            "windres failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        println!("cargo:rerun-if-env-changed=CC_SWITCH_EMBED_TEST_MANIFEST");
        if std::env::var_os("CC_SWITCH_EMBED_TEST_MANIFEST").is_some() {
            println!("cargo:rustc-link-arg={}", object_path.display());
        }
        println!("cargo:rerun-if-changed={}", manifest_path.display());
    }
}
