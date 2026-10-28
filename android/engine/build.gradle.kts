// Pure-Kotlin port of the WildFrame decision path (no Android dependencies), so it runs and is
// tested on a plain JVM. Can later become a Kotlin Multiplatform module for iOS.
plugins {
    id("org.jetbrains.kotlin.jvm")
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}

tasks.test {
    // golden/ lives at the repo root, one level above android/.
    systemProperty("golden.dir", rootProject.projectDir.resolve("../golden").canonicalPath)
    testLogging {
        events("failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        showStandardStreams = true
    }
}
