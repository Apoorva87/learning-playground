# Inside Prism · From screen to Swift

Start with the [interactive Apple TV screen](cases/swiftui-elements/runs/2026-11-24-teachable-code/index.html). Pick a movie poster. The code beside it shows how Prism puts artwork and a caption inside one selectable button. Move the focus right and press Select to see two different pieces of Swift respond.

| To understand | Where to look |
| --- | --- |
| How the app decides which page to display | Choose **Tab bar** for the six sections in `RootView.swift`. Each tab on the sketch also opens its own example. |
| How a catalog becomes a screen | Choose **Category rail**, **Title grid**, then **Poster and caption**. Each view builds the next piece. |
| How the remote changes what you see | Choose **Focus ring**, then open a title. `Focus.swift` draws the ring and `TitleDetailPage.swift` connects the Play button to an action. |
| Where data enters the picture | Choose **Search field**, **Title matching**, then **Results row**. Compare the words you type with the screen they produce. |

The Apple TV app starts in `RelayTV`. Its SwiftUI screens live in `RelayDesign/Sources/RelayUI`. The shared catalog, stores and search logic live in `RelayKit/Sources/RelayCore`. The project also has an iPad interface with its own layouts. This case follows the Apple TV version.

**Teachable Code trial, November 24, 2026.** The [guided code map](cases/swiftui-elements/runs/2026-11-24-teachable-code/walkthrough.html) covers fourteen elements and links each excerpt to the pinned source. The learner named Teachable Code among their three favorites so far. [Case brief](cases/swiftui-elements/brief.md) · [Trial notes](cases/swiftui-elements/runs/2026-11-24-teachable-code/notes.md) · [Tool setup](../../tools/teachable-code/README.md)
