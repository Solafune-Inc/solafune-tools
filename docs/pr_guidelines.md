# **Guidelines for Submitting Community Tools and Other Pull Requests either in solafune_tools.community_tools or improvement of other tools.**

1. **Accessibility**
    1. All submitted tools must be in a form that is easily accessible by other users within the solafune_tools.community_tools namespace.
    2. Clear documentation, including usage examples and any required dependencies, must be provided to ensure ease of adoption by the community.
2. **Structure and Design**
    1. Tools should follow functional or object-oriented programming paradigms.
    2. If a tool consists of multiple functionalities, it must be encapsulated within a class or module for better organization and maintainability.
    3. Tools should adhere to standard coding practices, including clear naming conventions, modularity, and maintainability. This ensures consistency and ease of understanding for future contributors.
3. **Language Compatibility**
    1. Tools can be developed in any programming language, provided they can be called and executed seamlessly from the solafune-tools Python implementation.
    2. If non-Python tools are used, a Python wrapper or bridge must be included, along with instructions for its integration.
4. **Originality and Compliance**
    1. Tools must be original code authored by the user. Assistance from generative AI is permitted but must be acknowledged in the tool’s documentation or comments.
    2. Direct copy-pasting from existing open-source software is not allowed.
    3. For tools inspired by or implementing concepts from papers or open-source software:
        1. Proper attribution and citation of the original work are mandatory.
        2. The submission must comply with the licensing terms of the original work.
5. **Quality Assurance**
    1. Tools must be thoroughly tested before submission.
    2. Unit tests and example scripts should be included to demonstrate functionality and verify reliability.
    3. Tools should handle errors gracefully and provide meaningful error messages to guide users in troubleshooting.
6. **Violation and Consequences**
    1. If a tool is found to be plagiarized or in violation of licensing terms:
        1. It will be removed from the project repository.
        2. If the user wins any award, prize money, or recognition (e.g., certificates, trophies), they must return these awards.
        3. The user will be disqualified from the award competition, and the award will be reassigned to a deserving participant.
    2. Repeated violations may result in a permanent ban from participating in future community projects or competitions.
7. **Collaboration and Community Engagement**
    1. Users are encouraged to collaborate with others in the community, leveraging shared knowledge and feedback to improve their tools.
    2. Tools should provide a clear contact point (e.g., GitHub repository, email) for reporting issues or suggesting improvements.
8. **Submission and Review Process**
    1. Submissions will undergo a review process by a panel of maintainers or selected community members to ensure compliance with the guidelines.
    2. Feedback on necessary changes or improvements will be provided during the review process.
    3. Approved tools will be merged into the official solafune_tools.community_tools repository.

Japanese Version

### Solafune-Tools について

Solafune-Tools は、研究と地理空間分析に不可欠なツールを提供することでリモート センシング コミュニティを強化するように設計されたオープン ソース ソフトウェア プロジェクトです。リモート センシング関連のタスクを容易にし、Solafune が主催するコンテストでのユーザー エクスペリエンスを向上させ、シームレスなデータ処理、分析、コラボレーションを実現します。

solafune_tools.community_tools またはその他のツールの改善に関するコミュニティ ツールおよびその他のプル リクエストを送信するためのガイドライン。

1. **アクセシビリティ**
    1. 提出されたすべてのツールは、solafune_tools.community_tools 名前空間内の他のユーザーが簡単にアクセスできる形式である必要があります。
    2. コミュニティによる採用を容易にするために、使用例や必要な依存関係を含む明確なドキュメントを提供する必要があります。
2. **構造と設計**
    1. ツールは、関数型またはオブジェクト指向プログラミング パラダイムに従う必要があります。
    2. ツールが複数の機能で構成されている場合は、整理と保守性を向上させるために、クラスまたはモジュール内にカプセル化する必要があります。
    3. ツールは、明確な命名規則、モジュール性、保守性など、標準的なコーディング プラクティスに準拠する必要があります。これにより、将来の貢献者にとって一貫性と理解しやすさが確保されます。
3. **言語の互換性**
    1. ツールは、solafune-tools Python 実装からシームレスに呼び出して実行できる限り、任意のプログラミング言語で開発できます。
    2. Python 以外のツールを使用する場合は、Python ラッパーまたはブリッジとその統合手順を含める必要があります。
4. **独創性とコンプライアンス**
    1. ツールは、ユーザーが作成したオリジナル コードである必要があります。生成 AI による支援は許可されますが、ツールのドキュメントまたはコメントでその旨を明記する必要があります。
    2. 既存のオープン ソース ソフトウェアからの直接コピー アンド ペーストは許可されません。
    3. 論文やオープン ソース ソフトウェアから着想を得た、または論文やオープン ソース ソフトウェアの概念を実装したツールの場合:
        1. 元の作品の適切な帰属と引用は必須です。
        2. 提出物は元の作品のライセンス条件に準拠する必要があります。
5. **品質保証**
    1. ツールは提出前に徹底的にテストする必要があります。
    2. 機能性を実証し、信頼性を検証するために、単体テストとサンプル スクリプトを含める必要があります。
    3. ツールはエラーを適切に処理し、トラブルシューティングでユーザーをガイドする意味のあるエラー メッセージを提供する必要があります。
6. **違反と結果**
    1. ツールが盗作またはライセンス条件に違反していることが判明した場合:
        1. プロジェクト リポジトリから削除されます。
        2. ユーザーが賞、賞金、または表彰 (証明書、トロフィーなど) を獲得した場合、これらの賞を返却する必要があります。
        3. ユーザーは賞のコンテストから失格となり、賞はふさわしい参加者に再割り当てされます。
    2. 違反を繰り返すと、今後のコミュニティ プロジェクトまたはコンテストへの参加が永久に禁止される場合があります。
7. **コラボレーションとコミュニティの関与**
    1. ユーザーはコミュニティ内の他のユーザーとコラボレーションし、共有された知識とフィードバックを活用してツールを改善することが推奨されます。
    2. ツールは、問題を報告したり改善を提案したりするための明確な連絡先 (GitHub リポジトリ、メールなど) を提供する必要があります。
8. **提出とレビューのプロセス**
    1. 提出されたツールは、ガイドラインに準拠していることを確認するために、メンテナーまたは選ばれたコミュニティ メンバーのパネルによるレビュー プロセスを受けます。
    2. レビュー プロセス中に、必要な変更や改善に関するフィードバックが提供されます。
    3. 承認されたツールは、公式の solafune_tools.community_tools リポジトリに統合されます。