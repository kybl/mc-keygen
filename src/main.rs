mod bench;
#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "metal")]
mod metal_gpu;
mod search;
mod simd4;
mod types;

use std::io::{self, stdout};
use std::time::Duration;

use clap::{Parser, Subcommand};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Gauge, Paragraph},
};

use std::sync::Arc;

use search::{SearchHandle, StreamHandle, Target};

#[derive(Parser)]
#[command(
    name = "mc-keygen",
    version,
    about = "MeshCore vanity Ed25519 key generator",
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Hex pattern(s) to match (0-9/A-F). Position is set by --where; with
    /// multiple patterns, any match wins.
    prefix: Vec<String>,

    /// Where in the key the pattern must appear.
    #[arg(long = "where", value_enum, default_value_t = WhereArg::Prefix)]
    location: WhereArg,

    /// Instead of an exact pattern, find a run of at least N identical hex
    /// characters (use with --where to pick prefix/anywhere/suffix).
    #[arg(long, value_name = "N", conflicts_with = "prefix")]
    run: Option<u32>,

    /// Number of worker threads (default: all cores)
    #[arg(short = 't', long = "threads")]
    threads: Option<usize>,

    /// Output result as JSON
    #[arg(long)]
    json: bool,

    /// Run forever: print every match to stdout (one line each) and keep
    /// searching instead of stopping at the first hit. Pipe to a file to
    /// collect matches over days, e.g. `mc-keygen ABC --stream > keys.txt`.
    #[arg(long)]
    stream: bool,

    /// Force CPU-only search (no GPU even if available)
    #[cfg(feature = "gpu")]
    #[arg(long, conflicts_with = "gpu_only")]
    cpu_only: bool,

    /// Force GPU-only search (no CPU threads)
    #[cfg(feature = "gpu")]
    #[arg(long, conflicts_with = "cpu_only")]
    gpu_only: bool,

    /// Verify GPU keygen matches host-side reference (run 64 chain steps and compare scalar/pubkey at each step)
    #[cfg(feature = "gpu")]
    #[arg(long)]
    verify: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Benchmark performance across CPU/GPU/hybrid modes and save JSONL records.
    Bench(bench::BenchArgs),
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum WhereArg {
    /// Match at the start of the key (default).
    Prefix,
    /// Match anywhere in the key.
    Anywhere,
    /// Match at the end of the key.
    Suffix,
}

impl From<WhereArg> for search::Location {
    fn from(w: WhereArg) -> Self {
        match w {
            WhereArg::Prefix => search::Location::Prefix,
            WhereArg::Anywhere => search::Location::Anywhere,
            WhereArg::Suffix => search::Location::Suffix,
        }
    }
}

/// Validate a hex pattern for the given location. The 00/FF leading-byte rule
/// and 62-char cap only apply to prefix search (which uses the sign-free fast
/// path); anywhere/suffix patterns may be up to 64 chars and any hex.
fn validate_pattern(pattern: &str, location: WhereArg) -> Result<String, String> {
    let upper = pattern.to_ascii_uppercase();
    let cap = 64;
    if upper.is_empty() || upper.len() > cap {
        return Err(format!(
            "pattern must be 1-{} hex characters, got {}",
            cap,
            upper.len()
        ));
    }
    if !upper.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("pattern must be valid hex (0-9, A-F), got '{}'", pattern));
    }
    if location == WhereArg::Prefix {
        return validate_prefix(&upper);
    }
    Ok(upper)
}

pub(crate) fn validate_prefix(prefix: &str) -> Result<String, String> {
    let upper = prefix.to_ascii_uppercase();

    // Cap is 62 not 64: nibble 63 lands on the high nibble of pubkey[31], which
    // contains the Ed25519 sign bit. The GPU kernel's fast path skips writing
    // that bit (saves an fe_mul per iter), so a prefix that reads byte 31
    // would compare against a zeroed sign bit and miss real matches.
    if upper.is_empty() || upper.len() > 62 {
        return Err(format!(
            "prefix must be 1-62 hex characters, got {} characters",
            upper.len()
        ));
    }

    if !upper.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("prefix must be valid hex (0-9, A-F), got '{}'", prefix));
    }

    // Reject prefixes that would always start with 00 or FF
    if upper.starts_with("00") || upper.starts_with("FF") {
        return Err(format!(
            "prefix '{}' starts with 00 or FF, which are skipped by MeshCore",
            upper
        ));
    }

    Ok(upper)
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        let mins = (secs / 60.0).floor();
        let rem = secs - mins * 60.0;
        format!("{}m {:.0}s", mins as u64, rem)
    } else {
        let hours = (secs / 3600.0).floor();
        let rem = secs - hours * 3600.0;
        let mins = (rem / 60.0).floor();
        format!("{}h {}m", hours as u64, mins as u64)
    }
}

/// Undo everything the TUI did to the terminal. ratatui hides the cursor on
/// every draw (`?25l`), and that outlives the alternate screen — without an
/// explicit show, the shell prompt comes back without a cursor.
fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen, crossterm::cursor::Show)?;
    Ok(())
}

fn run_tui_loop(
    handle: SearchHandle,
    search_desc: &str,
    expected: u64,
    mode_label: &str,
) -> io::Result<Result<types::SearchResult, types::SearchError>> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let prefix_display = search_desc.to_string();
    let prefix_label = "Searching for: ".to_string();

    let result = loop {
        let stats = handle.stats(expected);
        let done = handle.is_done();

        let mode_label_owned = mode_label.to_string();
        let prefix_label_owned = prefix_label.clone();
        let prefix_display_owned = prefix_display.clone();
        terminal.draw(|frame| {
            let area = frame.area();

            let outer = Block::default()
                .title(" mc-keygen ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan));

            let inner = outer.inner(area);
            frame.render_widget(outer, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(1), // prefix line
                    Constraint::Length(1), // blank
                    Constraint::Length(1), // gauge
                    Constraint::Length(1), // blank
                    Constraint::Length(1), // keys checked
                    Constraint::Length(1), // speed
                    Constraint::Length(1), // elapsed
                    Constraint::Length(1), // est remaining
                    Constraint::Min(0),   // spacer
                ])
                .split(inner);

            // Prefix line
            let prefix_line = Line::from(vec![
                Span::styled(&*prefix_label_owned, Style::default().fg(Color::Gray)),
                Span::styled(
                    &*prefix_display_owned,
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  ({})", mode_label_owned),
                    Style::default().fg(Color::DarkGray),
                ),
            ]);
            frame.render_widget(Paragraph::new(prefix_line), chunks[0]);

            // Progress gauge
            let exp = stats.expected_attempts;
            let ratio = if exp > 0 {
                (stats.attempts as f64 / exp as f64).min(1.0)
            } else {
                0.0
            };
            let pct_actual = if exp > 0 {
                stats.attempts as f64 / exp as f64 * 100.0
            } else {
                0.0
            };
            let gauge_label = format!(
                "{:.0}%  ({}/{})",
                pct_actual,
                format_number(stats.attempts),
                format_number(exp),
            );
            let gauge = Gauge::default()
                .gauge_style(Style::default().fg(Color::Green).bg(Color::DarkGray))
                .ratio(ratio)
                .label(gauge_label);
            frame.render_widget(gauge, chunks[2]);

            // Stats
            let keys_line = Line::from(vec![
                Span::styled("Keys checked:   ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format_number(stats.attempts),
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
            ]);
            frame.render_widget(Paragraph::new(keys_line), chunks[4]);

            let speed_line = Line::from(vec![
                Span::styled("Speed:          ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{} keys/sec", format_number(stats.keys_per_sec as u64)),
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
            ]);
            frame.render_widget(Paragraph::new(speed_line), chunks[5]);

            let elapsed_line = Line::from(vec![
                Span::styled("Elapsed:        ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format_duration(stats.elapsed_secs),
                    Style::default().fg(Color::White),
                ),
            ]);
            frame.render_widget(Paragraph::new(elapsed_line), chunks[6]);

            let remaining = if stats.keys_per_sec > 0.0 && stats.attempts < exp {
                let rem = (exp - stats.attempts) as f64 / stats.keys_per_sec;
                format_duration(rem)
            } else if stats.attempts >= exp {
                "any moment...".to_string()
            } else {
                "calculating...".to_string()
            };
            let remaining_line = Line::from(vec![
                Span::styled("Est. remaining: ", Style::default().fg(Color::Gray)),
                Span::styled(remaining, Style::default().fg(Color::White)),
            ]);
            frame.render_widget(Paragraph::new(remaining_line), chunks[7]);
        })?;

        if done {
            break handle.finish();
        }

        // Poll for Ctrl+C / 'q' to allow clean exit, otherwise tick every 50ms
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q')
                    || key.code == KeyCode::Char('c')
                        && key.modifiers.contains(event::KeyModifiers::CONTROL)
                {
                    // Restore terminal before exiting
                    restore_terminal()?;
                    std::process::exit(130);
                }
            }
        }
    };

    restore_terminal()?;

    Ok(result)
}

/// Run-forever streaming search: spawn workers and print every match to
/// stdout as it arrives, flushing each line so a redirected file stays current
/// even if the run is killed days later.
///
/// Non-JSON output is one tab-separated line per match:
///     <matched>\t<public_key>\t<private_key>
/// With `--json` each line is a standalone JSON object (JSON Lines).
fn run_stream(target: Arc<Target>, num_threads: usize, search_desc: &str, json: bool) {
    use std::io::Write;

    let handle = StreamHandle::start(target, num_threads);

    // Status goes to stderr so stdout carries only result data.
    eprintln!(
        "Streaming matches for {} on {} threads. Each hit is printed to stdout; press Ctrl+C to stop.",
        search_desc, num_threads
    );

    let stdout = io::stdout();
    while let Some(r) = handle.next_match() {
        let mut out = stdout.lock();
        let line = if json {
            serde_json::to_string(&r).unwrap()
        } else {
            format!("{}\t{}\t{}", r.matched_prefix, r.public_key, r.private_key)
        };
        // If stdout is gone (pipe closed), stop quietly.
        if writeln!(out, "{}", line).is_err() || out.flush().is_err() {
            break;
        }
    }
}

fn print_colored_result(result: &types::SearchResult) {
    use crossterm::style::{self, Stylize};

    // Green checkmark + bold "Match found!" line
    let attempts_str = format_number(result.attempts);
    let speed = if result.elapsed_secs > 0.0 {
        format_number((result.attempts as f64 / result.elapsed_secs) as u64)
    } else {
        "N/A".to_string()
    };

    eprintln!(
        "{}",
        style::style(format!(
            " ✓ Match found!  {} attempts in {} ({} keys/sec)",
            attempts_str,
            format_duration(result.elapsed_secs),
            speed,
        ))
        .green()
        .bold()
    );
    eprintln!();

    eprintln!(
        "{}{}",
        style::style("Matched:     ").dim(),
        style::style(&result.matched_prefix).yellow().bold()
    );

    // Public key, highlighting the matched hex pattern wherever it occurs.
    eprint!("{}", style::style("Public Key:  ").dim());
    let needle = result.matched_prefix.to_ascii_uppercase();
    if !needle.is_empty() {
        if let Some(pos) = result.public_key.find(&needle) {
            let (before, rest) = result.public_key.split_at(pos);
            let (matched, after) = rest.split_at(needle.len());
            eprint!("{}", style::style(before).white());
            eprint!("{}", style::style(matched).green().bold());
            eprintln!("{}", style::style(after).white());
        } else {
            eprintln!("{}", style::style(&result.public_key).white());
        }
    } else {
        eprintln!("{}", style::style(&result.public_key).white());
    }

    // Private key
    eprint!(
        "{}",
        style::style("Private Key: ").dim()
    );
    eprintln!(
        "{}",
        style::style(&result.private_key).white()
    );
}

fn print_colored_error(msg: &str) {
    use crossterm::style::{self, Stylize};
    eprintln!(
        "{}",
        style::style(format!(" ✗ Error: {}", msg)).red().bold()
    );
}

#[allow(unused_variables)]
pub(crate) fn try_init_gpu(prefixes: &[String]) -> Vec<Box<dyn search::GpuSearcher>> {
    #[cfg(feature = "metal")]
    {
        match metal_gpu::MetalSearcher::new(prefixes) {
            Ok(s) => return vec![Box::new(s)],
            Err(e) => {
                eprintln!("Warning: Metal GPU unavailable ({}), using CPU only", e);
                return vec![];
            }
        }
    }
    #[cfg(feature = "cuda")]
    {
        match gpu::CudaSearcher::new(prefixes) {
            Ok(s) => return vec![Box::new(s)],
            Err(e) => {
                eprintln!("Warning: CUDA GPU unavailable ({}), using CPU only", e);
                return vec![];
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        vec![]
    }
}

fn gpu_names_label(searchers: &[Box<dyn search::GpuSearcher>]) -> String {
    searchers
        .iter()
        .map(|g| g.device_name())
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let cli = Cli::parse();

    if let Some(Command::Bench(args)) = cli.command {
        if let Err(e) = bench::run(args) {
            eprintln!("bench failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    let report_error = |e: &str| {
        if cli.json {
            eprintln!("Error: {}", e);
        } else {
            print_colored_error(e);
        }
    };

    // Build the search target from --where / --run / patterns.
    let (target, what_desc) = if let Some(n) = cli.run {
        if !(1..=64).contains(&n) {
            report_error("--run N must be between 1 and 64");
            std::process::exit(1);
        }
        (
            Target::run(cli.location.into(), n),
            format!("run of {}+ identical chars", n),
        )
    } else {
        if cli.prefix.is_empty() {
            report_error("specify one or more hex patterns, or --run N");
            std::process::exit(1);
        }
        let mut patterns = Vec::new();
        for raw in &cli.prefix {
            match validate_pattern(raw, cli.location) {
                Ok(p) => patterns.push(p),
                Err(e) => {
                    report_error(&e);
                    std::process::exit(1);
                }
            }
        }
        let desc = patterns.join(", ");
        (Target::exact(cli.location.into(), &patterns), desc)
    };
    let target = Arc::new(target);

    let expected = target.expected_attempts();

    let where_desc = match cli.location {
        WhereArg::Prefix => "prefix",
        WhereArg::Anywhere => "anywhere",
        WhereArg::Suffix => "suffix",
    };
    let search_desc = format!("{} [{}]", what_desc, where_desc);

    // Run-forever streaming mode: CPU-only, no TUI. Print every match to
    // stdout as it's found and keep going until the process is killed.
    if cli.stream {
        let num_threads = cli.threads.unwrap_or_else(search::default_cpu_threads);
        run_stream(Arc::clone(&target), num_threads, &search_desc, cli.json);
        return;
    }

    #[cfg(feature = "gpu")]
    let cpu_only = cli.cpu_only;
    #[cfg(not(feature = "gpu"))]
    let cpu_only = true;

    #[cfg(feature = "gpu")]
    let gpu_only = cli.gpu_only;
    #[cfg(not(feature = "gpu"))]
    let gpu_only = false;

    #[cfg(feature = "gpu")]
    if cli.verify {
        eprint!("Compiling GPU kernel and running verification... ");
        #[cfg(feature = "cuda")]
        let result = gpu::verify_gpu_keygen().map_err(|e| format!("{}", e));
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        let result = metal_gpu::verify_gpu_keygen().map_err(|e| format!("{}", e));
        match result {
            Ok(()) => {
                eprintln!("PASSED");
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("FAILED: {}", e);
                std::process::exit(1);
            }
        }
    }

    // The GPU kernels only do prefix-exact search; other modes are CPU-only.
    let gpu_prefixes = target.gpu_prefixes();
    if gpu_only && gpu_prefixes.is_none() {
        report_error("--gpu-only only supports prefix search (not --where anywhere/suffix or --run)");
        std::process::exit(1);
    }
    let gpu_searchers = match (cpu_only, &gpu_prefixes) {
        (false, Some(p)) => try_init_gpu(p),
        _ => vec![],
    };

    // Hybrid mode reserves cores for the GPU dispatch thread; pure-CPU picks
    // a thread count from the SMT/hybrid topology. Explicit -t overrides either.
    let num_threads = cli.threads.unwrap_or_else(|| {
        if !gpu_searchers.is_empty() && !gpu_only {
            search::default_hybrid_cpu_threads(gpu_searchers.len())
        } else {
            search::default_cpu_threads()
        }
    });

    let (handle, mode_label) = if gpu_only {
        if gpu_searchers.is_empty() {
            report_error("--gpu-only requested but no GPU available");
            std::process::exit(1);
        }
        let label = gpu_names_label(&gpu_searchers);
        (SearchHandle::start_gpu(Arc::clone(&target), gpu_searchers), label)
    } else if gpu_searchers.is_empty() {
        let label = format!("{} threads", num_threads);
        (SearchHandle::start(Arc::clone(&target), num_threads), label)
    } else {
        let gpu_label = gpu_names_label(&gpu_searchers);
        let label = format!("{} + {} threads", gpu_label, num_threads);
        (
            SearchHandle::start_hybrid(Arc::clone(&target), num_threads, gpu_searchers),
            label,
        )
    };

    if cli.json {
        match handle.finish() {
            Ok(result) => println!("{}", serde_json::to_string_pretty(&result).unwrap()),
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        let search_result = match run_tui_loop(handle, &search_desc, expected, &mode_label) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("TUI error: {}", e);
                std::process::exit(1);
            }
        };
        match search_result {
            Ok(result) => print_colored_result(&result),
            Err(e) => {
                print_colored_error(&format!("{}", e));
                std::process::exit(1);
            }
        }
    }
}
