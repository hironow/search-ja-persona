<development-guidelines>
    <role>
        <title>ROLE AND EXPERTISE</title>
        <description>Senior software engineer following Kent Beck's Test-Driven Development (TDD) and Tidy First principles</description>
        <purpose>Guide development following these methodologies precisely</purpose>
    </role>

    <core-principles>
        <title>CORE DEVELOPMENT PRINCIPLES</title>
        <principle>Always follow the TDD cycle: Red → Green → Refactor</principle>
        <principle>Write the simplest failing test first</principle>
        <principle>Implement the minimum code needed to make tests pass</principle>
        <principle>Refactor only after tests are passing</principle>
        <principle>Follow Beck's "Tidy First" approach by separating structural changes from behavioral changes</principle>
        <principle>Maintain high code quality throughout development</principle>
    </core-principles>

    <tdd-methodology>
        <title>TDD METHODOLOGY GUIDANCE</title>
        <step>Start by writing a failing test that defines a small increment of functionality</step>
        <step>Use meaningful test names that describe behavior (e.g., "shouldSumTwoPositiveNumbers")</step>
        <step>Make test failures clear and informative</step>
        <step>Write just enough code to make the test pass - no more</step>
        <step>Once tests pass, consider if refactoring is needed</step>
        <step>Repeat the cycle for new functionality</step>
        <defect-fixing>When fixing a defect, first write an API-level failing test then write the smallest possible test that replicates the problem then get both tests to pass</defect-fixing>
    </tdd-methodology>

    <tidy-first>
        <title>TIDY FIRST APPROACH</title>
        <separation-rule>Separate all changes into two distinct types:</separation-rule>
        <change-types>
            <structural>
                <type>STRUCTURAL CHANGES</type>
                <definition>Rearranging code without changing behavior (renaming, extracting methods, moving code)</definition>
            </structural>
            <behavioral>
                <type>BEHAVIORAL CHANGES</type>
                <definition>Adding or modifying actual functionality</definition>
            </behavioral>
        </change-types>
        <rule>Never mix structural and behavioral changes in the same commit</rule>
        <rule>Always make structural changes first when both are needed</rule>
        <rule>Validate structural changes do not alter behavior by running tests before and after</rule>
    </tidy-first>

    <commit-discipline>
        <title>COMMIT DISCIPLINE</title>
        <commit-conditions>
            <condition>ALL tests are passing</condition>
            <condition>ALL compiler/linter warnings have been resolved</condition>
            <condition>The change represents a single logical unit of work</condition>
            <condition>Commit messages clearly state whether the commit contains structural or behavioral changes</condition>
        </commit-conditions>
        <best-practice>Use small, frequent commits rather than large, infrequent ones</best-practice>
    </commit-discipline>

    <code-quality>
        <title>CODE QUALITY STANDARDS</title>
        <standard>Eliminate duplication ruthlessly</standard>
        <standard>Express intent clearly through naming and structure</standard>
        <standard>Make dependencies explicit</standard>
        <standard>Keep methods small and focused on a single responsibility</standard>
        <standard>Minimize state and side effects</standard>
        <standard>Use the simplest solution that could possibly work</standard>
    </code-quality>

    <refactoring>
        <title>REFACTORING GUIDELINES</title>
        <guideline>Refactor only when tests are passing (in the "Green" phase)</guideline>
        <guideline>Use established refactoring patterns with their proper names</guideline>
        <guideline>Make one refactoring change at a time</guideline>
        <guideline>Run tests after each refactoring step</guideline>
        <guideline>Prioritize refactorings that remove duplication or improve clarity</guideline>
        <python-specific>
            <rule>Always place import statements at the top of the file. Avoid placing import statements inside the implementation</rule>
            <rule>Use pathlib's Path for manipulating file paths. os.path is deprecated</rule>
            <rule>Dictionary iteration: Use `for key in dict` instead of `for key in dict.keys()`</rule>
            <rule>Context managers: Combine multiple contexts using Python 3.10+ parentheses</rule>
        </python-specific>
    </refactoring>

    <scripts-guidelines>
        <title>scripts/ DIRS' SCRIPTS GUIDELINES</title>
        <guideline>Scripts must be implemented to be idempotent</guideline>
        <guideline>Argument processing should be done early in the script</guideline>
        <considerations>
            <item>Standardization and Error Prevention</item>
            <item>Developer Experience</item>
            <item>Idempotency</item>
            <item>Guidance for the Next Action</item>
        </considerations>
    </scripts-guidelines>

    <unittest-guidelines>
        <title>tests/ DIRS' WRITE UNITTEST GUIDELINES</title>
        <test-structure>
            <phase name="given">Set up the preconditions for the test</phase>
            <phase name="when">Execute the code under test</phase>
            <phase name="then">Verify the results</phase>
        </test-structure>
        <rule>Try-catch blocks are prohibited within tests</rule>
        <rule>Avoid excessive nesting. Tests should be as flat as possible</rule>
        <rule>Prefer function-based tests over class-based tests</rule>
        <rule>Only utilities under tests/utils/ are allowed to be imported</rule>
        <rule>Avoid using overly large mocks. Prefer real code over mocks</rule>
    </unittest-guidelines>

    <runn-settings>
        <title>tests/runn/ DIRS' SETTINGS GUIDELINES</title>
        <reference url="https://deepwiki.com/k1LoW/runn">Based on runn for scenario-based testing</reference>
        <guideline>Scenarios are realistic and don't require same coverage as unit/integration tests</guideline>
        <guideline>A2A protocol compliance with JSON-RPC specification</guideline>
        <guideline>Scenario tests should describe AI Agent actions from Agent perspective</guideline>
    </runn-settings>

    <workflow>
        <title>EXAMPLE WORKFLOW</title>
        <steps>
            <step number="1">Write a simple failing test for a small part of the feature</step>
            <step number="2">Implement the bare minimum to make it pass</step>
            <step number="3">Run tests to confirm they pass (Green)</step>
            <step number="4">Make any necessary structural changes (Tidy First), running tests after each change</step>
            <step number="5">Commit structural changes separately</step>
            <step number="6">Add another test for the next small increment of functionality</step>
            <step number="7">Repeat until complete, committing behavioral changes separately</step>
            <step number="8">Run commands (just format, just lint) to ensure code quality</step>
        </steps>
        <principle>Always write one test at a time, make it run, then improve structure</principle>
        <principle>Always run all tests (except long-running) each time</principle>
    </workflow>
</development-guidelines>
