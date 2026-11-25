# Teachable Code on Prism’s Apple TV interface

November 24, 2026 · [Screen explorer](index.html) · [Guided map](walkthrough.html) · [Original request](../../brief.md)

## What we ran

We used Teachable Code at `e74598d4b0d2a13cf67182ec897ad8f308894495` with a clean, local copy of Prism at `960198d0b364f60eecd9cad591b2873b4cf51e58`. The classifier guessed “library”. Prism has an Apple TV application entry point and SwiftUI pages, so we selected its 2D app lesson. We wrote the [lesson source](lesson.js) for someone learning both Swift and UI design. The upstream builder verified fourteen citations and produced the guided map. Its controls step through the lesson, open source excerpts and offer prediction questions. [Setup and rebuild instructions](../../../../../../tools/teachable-code/README.md)

The playground adds a clickable teaching sketch and simulated remote. Its [build script](build.py) reads the builder’s checked excerpts so both views show the same Swift. The interface layouts are source-based sketches. The Apple TV app, iPad interface and live playback were outside this run.

## Evidence

The following commands run in the playground root after setup. The search-ranking test passed with one XCTest and zero failures. Its fixture merges two versions into the best title, Interstellar, with The Science of Interstellar leading the remaining titles. The separate driver called the real matcher with three example queries. Against Interstellar, `in` returned prefix, `INTERSTELLAR` returned exact, and `dune` returned no match. [Recorded matcher data](../../../../assets/match-output.json)

```sh
tool=.local/tools/teachable-code
assets=topics/inside-prism/assets
swift test --package-path "$tool/prism/RelayKit" --scratch-path "$tool/swift-build" --filter SearchEngineTests.testExactTitleRankingAndMerge
swiftc -module-cache-path "$tool/module-cache" "$tool/prism/RelayKit/Sources/RelayCore/Services/TextMatch.swift" "$assets/match-probe.swift" -o "$tool/match-probe"
"$tool/match-probe" > "$assets/match-output.json"
```

## Agent observations

Browse currently routes into categories. Selecting a movie in the poster grid opens its detail screen. The caption, focus ring and action live in different pieces of code, which makes one poster a good way to introduce composition and state. Tracing the app from RootView mattered here. BrowsePage has an older rows screen, but its current switch selects BrowseCategoriesScreen. The builder checks the quoted lines against the source. Following the route establishes which view the learner actually sees.

We checked the poster, focus movement, title details and Library in Safari, then opened the guided map and navigated to its final Settings step. The upstream automated layout checker could not start its Chromium process under the local sandbox. Phone-width rendering was not verified by that checker.

## Learner feedback

**November 24, 2026.** The learner named Teachable Code among their three favorites so far. After exploring a poster, which view helped you find the code for its caption and its remote action? Which would you keep for the next unfamiliar app?
