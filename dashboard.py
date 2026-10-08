import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stack Overflow Salary Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM THEME / CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy: #0f172a;
        --navy-2: #1e293b;
        --teal: #14b8a6;
        --cyan: #06b6d4;
        --violet: #8b5cf6;
        --amber: #f59e0b;
        --muted: #64748b;
        --line: rgba(100,116,139,.20);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(20,184,166,.08), transparent 24rem),
            radial-gradient(circle at 95% 3%, rgba(139,92,246,.07), transparent 28rem);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }

    .hero {
        padding: 2.1rem 2.25rem;
        border-radius: 24px;
        color: white;
        background:
            linear-gradient(125deg, rgba(15,23,42,.98), rgba(30,41,59,.96) 55%, rgba(20,184,166,.88));
        box-shadow: 0 20px 50px rgba(15,23,42,.18);
        margin-bottom: 1.35rem;
        overflow: hidden;
        position: relative;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        border-radius: 999px;
        right: -70px;
        top: -100px;
        border: 34px solid rgba(255,255,255,.08);
    }

    .hero-kicker {
        font-size: .78rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        font-weight: 700;
        opacity: .78;
        margin-bottom: .6rem;
    }

    .hero h1 {
        color: white !important;
        font-size: 2.55rem;
        line-height: 1.06;
        margin: 0 0 .8rem 0;
        letter-spacing: -.03em;
    }

    .hero p {
        font-size: 1.03rem;
        max-width: 900px;
        opacity: .90;
        line-height: 1.65;
        margin: 0;
    }

    .section-eyebrow {
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-weight: 800;
        color: var(--teal);
        margin-top: .6rem;
        margin-bottom: .25rem;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -.025em;
        margin-bottom: .35rem;
        line-height: 1.15;
    }

    .section-copy {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
        margin-bottom: 1.25rem;
        max-width: 980px;
    }

    .insight-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        background: rgba(255,255,255,.72);
        box-shadow: 0 8px 24px rgba(15,23,42,.05);
        min-height: 132px;
    }

    @media (prefers-color-scheme: dark) {
        .insight-card { background: rgba(30,41,59,.50); }
    }

    .insight-label {
        text-transform: uppercase;
        letter-spacing: .10em;
        font-size: .70rem;
        font-weight: 800;
        color: var(--muted);
        margin-bottom: .45rem;
    }

    .insight-value {
        font-size: 1.55rem;
        line-height: 1.15;
        font-weight: 800;
        margin-bottom: .4rem;
    }

    .insight-note {
        font-size: .88rem;
        color: var(--muted);
        line-height: 1.4;
    }

    .callout {
        border-left: 4px solid var(--teal);
        background: rgba(20,184,166,.08);
        border-radius: 0 14px 14px 0;
        padding: .9rem 1rem;
        margin: .8rem 0 1.2rem 0;
        line-height: 1.55;
    }

    .method-note {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.55;
        padding: .65rem .8rem;
        border-radius: 12px;
        border: 1px dashed var(--line);
        margin-top: .7rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1rem .85rem;
        background: rgba(255,255,255,.65);
        box-shadow: 0 8px 24px rgba(15,23,42,.04);
    }

    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] { background: rgba(30,41,59,.45); }
    }

    div[data-testid="stMetricValue"] {
        font-weight: 800;
        letter-spacing: -.02em;
    }

    div[data-testid="stPlotlyChart"], div[data-testid="stImage"], .stPyplot {
        border-radius: 18px;
        overflow: hidden;
    }

    hr {
        border: none;
        border-top: 1px solid var(--line);
        margin: 2rem 0;
    }

    .footer {
        color: var(--muted);
        font-size: .82rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("clean_stackoverflow_data.csv")


def money(x):
    return f"${x:,.0f}"


def salary_formatter(y, _):
    if y >= 1_000_000:
        return f"${y/1_000_000:.0f}M"
    if y >= 1_000:
        return f"${y/1_000:.0f}K"
    return f"${y:,.0f}"


def remove_high_iqr_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    return data[data <= upper_bound]


def section_intro(number, title, copy):
    st.markdown(
        f"""
        <div class="section-eyebrow">Section {number}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def callout(text):
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=.16, linewidth=.7)
    ax.set_axisbelow(True)


# ---------------------------------------------------------
# DATA + GLOBAL CALCULATIONS
# ---------------------------------------------------------
df = load_data()

# Question 1
jobsat_data = df[df['JobSat'].between(0, 10)].copy()
jobsat_summary = (
    jobsat_data.groupby('DevType')['JobSat']
    .agg(['mean', 'count'])
    .sort_values('mean', ascending=False)
)
jobsat_plot_data = jobsat_summary[jobsat_summary['count'] >= 20]

salary_summary = (
    df.groupby('DevType')['ConvertedCompYearly']
    .agg(['mean', 'count'])
    .sort_values('mean', ascending=False)
)
salary_plot_data = salary_summary[salary_summary['count'] >= 20]
age_salary_plot_data = df[df['Age'] <= 100].copy()

# Question 2
hybrid_salary = df.loc[df['RemoteWork'] == 'Hybrid', 'ConvertedCompYearly'].dropna()
remote_salary = df.loc[df['RemoteWork'] == 'Remote', 'ConvertedCompYearly'].dropna()
hybrid_salary_clean = remove_high_iqr_outliers(hybrid_salary)
remote_salary_clean = remove_high_iqr_outliers(remote_salary)

n_hybrid = len(hybrid_salary_clean)
n_remote = len(remote_salary_clean)
mean_hybrid = hybrid_salary_clean.mean()
mean_remote = remote_salary_clean.mean()
var_hybrid = hybrid_salary_clean.var(ddof=1)
var_remote = remote_salary_clean.var(ddof=1)
pooled_variance = (((n_hybrid - 1) * var_hybrid) + ((n_remote - 1) * var_remote)) / (n_hybrid + n_remote - 2)
pooled_std = np.sqrt(pooled_variance)
standard_error = pooled_std * np.sqrt(1 / n_hybrid + 1 / n_remote)
t_stat_manual = (mean_hybrid - mean_remote) / standard_error
df_t = n_hybrid + n_remote - 2
p_value_manual = 2 * stats.t.sf(abs(t_stat_manual), df=df_t)
t_stat_builtin, p_value_builtin = stats.ttest_ind(hybrid_salary_clean, remote_salary_clean, equal_var=True)
t_stat_welch, p_value_welch = stats.ttest_ind(hybrid_salary_clean, remote_salary_clean, equal_var=False)

@st.cache_data
def bootstrap_remote_hybrid(hybrid_values, remote_values, n_bootstrap=10000, seed=42):
    rng = np.random.default_rng(seed)
    hybrid_boot_means = np.empty(n_bootstrap)
    remote_boot_means = np.empty(n_bootstrap)
    hybrid_boot_medians = np.empty(n_bootstrap)
    remote_boot_medians = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        hybrid_sample = rng.choice(hybrid_values, size=len(hybrid_values), replace=True)
        remote_sample = rng.choice(remote_values, size=len(remote_values), replace=True)
        hybrid_boot_means[i] = hybrid_sample.mean()
        remote_boot_means[i] = remote_sample.mean()
        hybrid_boot_medians[i] = np.median(hybrid_sample)
        remote_boot_medians[i] = np.median(remote_sample)

    return hybrid_boot_means, remote_boot_means, hybrid_boot_medians, remote_boot_medians

hybrid_boot_means, remote_boot_means, hybrid_boot_medians, remote_boot_medians = bootstrap_remote_hybrid(
    hybrid_salary_clean.to_numpy(), remote_salary_clean.to_numpy()
)
boot_mean_diff = remote_boot_means - hybrid_boot_means
standardized_diff = (boot_mean_diff - boot_mean_diff.mean()) / boot_mean_diff.std(ddof=1)

mean_diff = remote_salary_clean.mean() - hybrid_salary_clean.mean()
se_diff = pooled_std * np.sqrt(1 / len(hybrid_salary_clean) + 1 / len(remote_salary_clean))
df_ci = len(hybrid_salary_clean) + len(remote_salary_clean) - 2
t_critical = stats.t.ppf(0.975, df_ci)
ci_lower_formula = mean_diff - t_critical * se_diff
ci_upper_formula = mean_diff + t_critical * se_diff
ci_lower_boot = np.percentile(boot_mean_diff, 2.5)
ci_upper_boot = np.percentile(boot_mean_diff, 97.5)

mood_stat, mood_p, grand_median, contingency_table = stats.median_test(hybrid_salary_clean, remote_salary_clean)
boot_median_diff = remote_boot_medians - hybrid_boot_medians
count_lower = np.sum(boot_median_diff <= 0)
count_upper = np.sum(boot_median_diff >= 0)
bootstrap_median_p = 2 * (min(count_lower, count_upper) + 1) / (10000 + 1)

# Question 3
north_america = df[df['Country'].isin(['United States of America', 'Canada'])].copy()
education_data = north_america[
    north_america['EdLevel'].isin(["Bachelor's degree", "Master's degree", "Professional degree"])
].copy()
education_groups = ["Bachelor's degree", "Master's degree", "Professional degree"]

cleaned_education_groups = []
for level in education_groups:
    group_data = education_data[education_data['EdLevel'] == level].copy()
    q1 = group_data['ConvertedCompYearly'].quantile(0.25)
    q3 = group_data['ConvertedCompYearly'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    group_data = group_data[group_data['ConvertedCompYearly'] <= upper_bound]
    cleaned_education_groups.append(group_data)

education_data_clean = pd.concat(cleaned_education_groups)
bachelors_salary = education_data_clean.loc[education_data_clean['EdLevel'] == "Bachelor's degree", 'ConvertedCompYearly']
masters_salary = education_data_clean.loc[education_data_clean['EdLevel'] == "Master's degree", 'ConvertedCompYearly']
professional_salary = education_data_clean.loc[education_data_clean['EdLevel'] == "Professional degree", 'ConvertedCompYearly']

f_stat, anova_p = stats.f_oneway(bachelors_salary, masters_salary, professional_salary)

@st.cache_data
def bootstrap_education(bachelor_values, master_values, professional_values, n_bootstrap=10000, seed=42):
    rng = np.random.default_rng(seed)
    bachelor_means = np.empty(n_bootstrap)
    master_means = np.empty(n_bootstrap)
    professional_means = np.empty(n_bootstrap)
    f_stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        b = rng.choice(bachelor_values, size=len(bachelor_values), replace=True)
        m = rng.choice(master_values, size=len(master_values), replace=True)
        p = rng.choice(professional_values, size=len(professional_values), replace=True)
        bachelor_means[i] = b.mean()
        master_means[i] = m.mean()
        professional_means[i] = p.mean()
        f_stats[i], _ = stats.f_oneway(b, m, p)

    return bachelor_means, master_means, professional_means, f_stats

bachelors_boot_means, masters_boot_means, professional_boot_means, bootstrap_f_stats = bootstrap_education(
    bachelors_salary.to_numpy(), masters_salary.to_numpy(), professional_salary.to_numpy()
)

diff_masters_bachelors = masters_boot_means - bachelors_boot_means
diff_professional_bachelors = professional_boot_means - bachelors_boot_means
diff_professional_masters = professional_boot_means - masters_boot_means

std_diff_masters_bachelors = (diff_masters_bachelors - diff_masters_bachelors.mean()) / diff_masters_bachelors.std(ddof=1)
std_diff_professional_bachelors = (diff_professional_bachelors - diff_professional_bachelors.mean()) / diff_professional_bachelors.std(ddof=1)
std_diff_professional_masters = (diff_professional_masters - diff_professional_masters.mean()) / diff_professional_masters.std(ddof=1)

k = 3
N = len(bachelors_salary) + len(masters_salary) + len(professional_salary)
df_between = k - 1
df_within = N - k
f_critical = stats.f.ppf(0.95, df_between, df_within)
bootstrap_significant = np.mean(bootstrap_f_stats > f_critical)

# ---------------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 MIE1624")
    st.caption("Assignment 1 · Client Dashboard")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "1 · Exploratory Analysis",
            "2 · Remote vs Hybrid",
            "3 · Education & Salary",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("2025 Stack Overflow Developer Survey")
    st.caption("Salary values shown in USD")

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">MIE1624 · Assignment 1</div>
        <h1>Stack Overflow Developer Salary Analysis</h1>
        <p>
            A client-facing view of how work mode and education relate to developer compensation,
            supported by exploratory analysis, hypothesis testing, bootstrapping and ANOVA.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    section_intro("00", "Executive Summary", "The strongest takeaways from the full analysis, translated into client-friendly language.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">Remote salary premium</div>
            <div class="insight-value">{money(mean_diff)}</div>
            <div class="insight-note">Average remote salary minus average hybrid salary after outlier treatment.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">Remote mean salary</div>
            <div class="insight-value">{money(mean_remote)}</div>
            <div class="insight-note">Compared with {money(mean_hybrid)} for hybrid workers.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">Education ANOVA</div>
            <div class="insight-value">F = {f_stat:.2f}</div>
            <div class="insight-note">p = {anova_p:.2e}; at least one education group differs significantly.</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">Bootstrap stability</div>
            <div class="insight-value">{bootstrap_significant*100:.0f}%</div>
            <div class="insight-note">Share of education bootstrap ANOVAs above the 5% critical F threshold.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Key findings")
    left, right = st.columns(2)
    with left:
        callout("<b>Work mode:</b> Remote workers earned a higher mean and median salary than hybrid workers, and the difference remained highly significant across parametric and bootstrap analyses.")
        callout("<b>Confidence intervals:</b> Both the formula-based and bootstrap 95% intervals for Remote − Hybrid stayed entirely above zero.")
    with right:
        callout("<b>Education:</b> Bachelor's degree holders had the lowest cleaned mean salary, while Master's and Professional degree holders were higher and similar to one another.")
        callout("<b>Robustness:</b> All 10,000 education bootstrap ANOVAs exceeded the critical F-value, reinforcing the original ANOVA result.")

    st.markdown("### Analysis roadmap")
    st.write("Use the sidebar to explore the three analysis sections. Each page contains the exact calculations and conclusions used in the assignment notebook, presented in a more client-friendly format.")

# ---------------------------------------------------------
# SECTION 1
# ---------------------------------------------------------
elif page == "1 · Exploratory Analysis":
    section_intro("01", "Exploratory Data Analysis", "Job satisfaction, compensation by role, and the relationship between age and salary.")

    t1, t2, t3 = st.tabs(["Job satisfaction", "Salary by job title", "Salary vs age"])

    with t1:
        st.markdown("### Average Job Satisfaction by Job Title")
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        jobsat_plot_data['mean'].sort_values().plot(kind='barh', ax=ax1, color='#14b8a6')
        ax1.set_xlabel('Average Job Satisfaction')
        ax1.set_ylabel('Job Title')
        ax1.set_xlim(0, 10)
        style_ax(ax1)
        st.pyplot(fig1, use_container_width=True)
        callout("<b>Finding:</b> Senior executives reported the highest average job satisfaction, while support engineers or analysts reported the lowest.")
        st.markdown('<div class="method-note">Method note: job titles with fewer than 20 valid responses were excluded because very small groups can produce unstable averages.</div>', unsafe_allow_html=True)

    with t2:
        st.markdown("### Mean Annual Salary by Job Title")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        salary_plot_data['mean'].sort_values().plot(kind='barh', ax=ax2, color='#8b5cf6')
        ax2.set_xlabel('Mean Annual Salary (USD)')
        ax2.set_ylabel('Job Title')
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        style_ax(ax2)
        st.pyplot(fig2, use_container_width=True)
        callout("<b>Anomaly:</b> Financial analysts or engineers have an unusually high mean salary, suggesting that extreme salary observations are inflating the average for this role.")

    with t3:
        st.markdown("### Annual Salary vs. Age")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(age_salary_plot_data['Age'], age_salary_plot_data['ConvertedCompYearly'], alpha=.22, s=18, color='#06b6d4')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Annual Salary (USD, Log Scale)')
        ax3.set_yscale('log')
        ax3.yaxis.set_major_formatter(FuncFormatter(salary_formatter))
        style_ax(ax3)
        st.pyplot(fig3, use_container_width=True)
        callout("<b>Finding:</b> Salary generally rises from younger ages into middle age, but the relationship is highly variable and contains several extreme high-salary outliers.")
        st.markdown('<div class="method-note">Method note: ages above 100 were excluded from this visualization; salary observations were retained and displayed on a logarithmic scale.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 2
# ---------------------------------------------------------
elif page == "2 · Remote vs Hybrid":
    section_intro("02", "Hybrid vs. Remote Salary Comparison", "A multi-method comparison of compensation across work modes using descriptive statistics, t-tests, bootstrap inference, confidence intervals and medians.")

    st.markdown("### Salary snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hybrid mean", money(mean_hybrid))
    c2.metric("Remote mean", money(mean_remote), delta=money(mean_diff))
    c3.metric("Hybrid median", money(hybrid_salary_clean.median()))
    c4.metric("Remote median", money(remote_salary_clean.median()), delta=money(remote_salary_clean.median()-hybrid_salary_clean.median()))
    callout("<b>Bottom line:</b> After high-end IQR outlier removal, remote workers had both a higher mean and a higher median salary than hybrid workers.")

    test_tab, boot_tab, ci_tab, median_tab = st.tabs(["Hypothesis tests", "Bootstrap", "Confidence intervals", "Median test"])

    with test_tab:
        st.markdown("### Two-sample t-tests")
        a, b, c = st.columns(3)
        a.metric("Manual pooled t", f"{t_stat_manual:.2f}")
        b.metric("Built-in pooled t", f"{t_stat_builtin:.2f}")
        c.metric("Welch t", f"{t_stat_welch:.2f}")
        st.dataframe(
            pd.DataFrame({
                "Method": ["Manual pooled", "Built-in pooled", "Welch"],
                "t-statistic": [t_stat_manual, t_stat_builtin, t_stat_welch],
                "p-value": [p_value_manual, p_value_builtin, p_value_welch],
            }).style.format({"t-statistic": "{:.3f}", "p-value": "{:.3e}"}),
            use_container_width=True,
            hide_index=True,
        )
        callout("<b>Conclusion:</b> All tests produce extremely small p-values, providing strong evidence that mean salaries differ between hybrid and remote workers.")

    with boot_tab:
        st.markdown("### Bootstrap distributions")
        f1, a1 = plt.subplots(figsize=(10, 5.5))
        a1.hist(hybrid_boot_means, bins=50, alpha=.62, label='Hybrid', color='#14b8a6')
        a1.hist(remote_boot_means, bins=50, alpha=.58, label='Remote', color='#8b5cf6')
        a1.set_xlabel('Bootstrapped Mean Salary (USD)')
        a1.set_ylabel('Frequency')
        a1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        a1.legend(frameon=False)
        style_ax(a1)
        st.pyplot(f1, use_container_width=True)

        f2, a2 = plt.subplots(figsize=(10, 5.2))
        a2.hist(boot_mean_diff, bins=50, alpha=.78, color='#06b6d4')
        a2.axvline(0, linestyle='--', color='#0f172a', label='No difference')
        a2.set_xlabel('Remote Mean − Hybrid Mean (USD)')
        a2.set_ylabel('Frequency')
        a2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        a2.legend(frameon=False)
        style_ax(a2)
        st.pyplot(f2, use_container_width=True)
        callout(f"<b>Bootstrap finding:</b> The mean difference distribution is centered around {money(boot_mean_diff.mean())} and lies well above zero.")

        with st.expander("Show standardized bootstrap distribution"):
            f3, a3 = plt.subplots(figsize=(10, 4.6))
            a3.hist(standardized_diff, bins=50, alpha=.78, color='#f59e0b')
            a3.axvline(0, linestyle='--', color='#0f172a')
            a3.set_xlabel('Standardized Difference in Means')
            a3.set_ylabel('Frequency')
            style_ax(a3)
            st.pyplot(f3, use_container_width=True)

    with ci_tab:
        st.markdown("### 95% confidence intervals for Remote − Hybrid")
        a, b = st.columns(2)
        a.metric("Formula-based 95% CI", f"{money(ci_lower_formula)} to {money(ci_upper_formula)}")
        b.metric("Bootstrap 95% CI", f"{money(ci_lower_boot)} to {money(ci_upper_boot)}")
        callout("<b>Interpretation:</b> Both intervals lie entirely above zero and are nearly identical, supporting a statistically higher mean salary for remote workers.")

    with median_tab:
        st.markdown("### Median salary comparison")
        a, b, c = st.columns(3)
        a.metric("Mood's p-value", f"{mood_p:.2e}")
        b.metric("Bootstrap median p", f"{bootstrap_median_p:.4f}")
        c.metric("Avg. median difference", money(boot_median_diff.mean()))
        callout("<b>Conclusion:</b> The numerical p-values differ in magnitude, but both methods indicate a statistically significant median salary difference.")

# ---------------------------------------------------------
# SECTION 3
# ---------------------------------------------------------
elif page == "3 · Education & Salary":
    section_intro("03", "Education Level and Salary", "A North America–only comparison of Bachelor's, Master's and Professional degree holders using ANOVA and bootstrap analysis.")

    st.markdown("### Education salary snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bachelor's mean", money(bachelors_salary.mean()))
    c2.metric("Master's mean", money(masters_salary.mean()), delta=money(masters_salary.mean()-bachelors_salary.mean()))
    c3.metric("Professional mean", money(professional_salary.mean()), delta=money(professional_salary.mean()-bachelors_salary.mean()))
    callout("<b>Finding:</b> Bachelor's degree holders had the lowest cleaned mean salary; Master's and Professional degree holders were higher and close to one another.")

    anova_tab, dist_tab, pair_tab, stability_tab = st.tabs(["ANOVA", "Bootstrap means", "Pairwise differences", "Stability"])

    with anova_tab:
        st.markdown("### One-way ANOVA")
        a, b = st.columns(2)
        a.metric("F-statistic", f"{f_stat:.2f}")
        b.metric("p-value", f"{anova_p:.2e}")
        callout("<b>Conclusion:</b> The p-value is far below 0.05, so at least one of the three education groups has a significantly different mean salary.")

    with dist_tab:
        st.markdown("### Bootstrap mean salary distributions")
        f7, a7 = plt.subplots(figsize=(10, 5.7))
        a7.hist(bachelors_boot_means, bins=50, alpha=.55, label="Bachelor's", color='#06b6d4')
        a7.hist(masters_boot_means, bins=50, alpha=.52, label="Master's", color='#8b5cf6')
        a7.hist(professional_boot_means, bins=50, alpha=.48, label='Professional', color='#14b8a6')
        a7.set_xlabel('Bootstrapped Mean Salary (USD)')
        a7.set_ylabel('Frequency')
        a7.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        a7.legend(frameon=False)
        style_ax(a7)
        st.pyplot(f7, use_container_width=True)
        callout("<b>Pattern:</b> The Bachelor's distribution is noticeably lower, while the Master's and Professional distributions overlap substantially.")

    with pair_tab:
        st.markdown("### Pairwise differences in mean salary")
        f8, a8 = plt.subplots(figsize=(10, 5.7))
        a8.hist(diff_masters_bachelors, bins=50, alpha=.54, label="Master's vs Bachelor's", color='#8b5cf6')
        a8.hist(diff_professional_bachelors, bins=50, alpha=.52, label="Professional vs Bachelor's", color='#14b8a6')
        a8.hist(diff_professional_masters, bins=50, alpha=.48, label="Professional vs Master's", color='#f59e0b')
        a8.axvline(0, linestyle='--', color='#0f172a', label='No difference')
        a8.set_xlabel('Difference in Mean Salary (USD)')
        a8.set_ylabel('Frequency')
        a8.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        a8.legend(frameon=False)
        style_ax(a8)
        st.pyplot(f8, use_container_width=True)
        callout("<b>Pairwise view:</b> Master's vs Bachelor's and Professional vs Bachelor's are centered above zero; Professional vs Master's is centered close to zero.")

        with st.expander("Show standardized pairwise differences"):
            f9, a9 = plt.subplots(figsize=(10, 4.8))
            a9.hist(std_diff_masters_bachelors, bins=50, alpha=.50, label="Master's vs Bachelor's", color='#8b5cf6')
            a9.hist(std_diff_professional_bachelors, bins=50, alpha=.48, label="Professional vs Bachelor's", color='#14b8a6')
            a9.hist(std_diff_professional_masters, bins=50, alpha=.46, label="Professional vs Master's", color='#f59e0b')
            a9.axvline(0, linestyle='--', color='#0f172a')
            a9.set_xlabel('Standardized Difference in Means')
            a9.set_ylabel('Frequency')
            a9.legend(frameon=False)
            style_ax(a9)
            st.pyplot(f9, use_container_width=True)

    with stability_tab:
        st.markdown("### Bootstrap ANOVA stability")
        a, b, c = st.columns(3)
        a.metric("Original F-statistic", f"{f_stat:.2f}")
        b.metric("Critical F-value", f"{f_critical:.2f}")
        c.metric("Bootstrap significant", f"{bootstrap_significant*100:.0f}%")
        callout("<b>Robustness:</b> All 10,000 bootstrap ANOVA replications exceeded the 5% critical F threshold, supporting the same conclusion as the original ANOVA.")

st.markdown('<div class="footer">Analysis based on the 2025 Stack Overflow Developer Survey · MIE1624 Assignment 1</div>', unsafe_allow_html=True)