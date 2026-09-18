import java.io.File
import org.apache.tools.ant.taskdefs.condition.Os
import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.logging.LogLevel
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.TaskAction

/**
 * 定制（M5）：不再经 `node tauri android android-studio-script`（该子命令依赖
 * Android Studio 的 WebSocket dev server），直接调用 cargo 构建 cdylib。
 * 交叉编译环境变量（linker/CC/AR）由调用方 shell 或 CI 注入。
 */
open class BuildTask : DefaultTask() {
    @Input
    var rootDirRel: String? = null
    @Input
    var target: String? = null
    @Input
    var release: Boolean? = null

    @TaskAction
    fun assemble() {
        val rootDirRel = rootDirRel ?: throw GradleException("rootDirRel cannot be null")
        val target = target ?: throw GradleException("target cannot be null")
        val release = release ?: throw GradleException("release cannot be null")

        val profile = if (release) "release" else "debug"
        val projectDir = File(project.projectDir, rootDirRel)
        // Android ABI 名 → Rust 目标三元组
        val rustTarget = when (target) {
            "arm64", "aarch64" -> "aarch64-linux-android"
            "armv7" -> "armv7-linux-androideabi"
            "x86", "i686" -> "i686-linux-android"
            "x86_64" -> "x86_64-linux-android"
            else -> target
        }

        project.exec {
            workingDir(projectDir)
            executable("cargo")
            args(
                listOf(
                    "build",
                    "--package", "gscp-app",
                    "--lib",
                    "--target", rustTarget,
                )
            )
            if (release) {
                args("--release")
            }
        }.assertNormalExitValue()
    }
}
