#include "foye_fastmath_fp32.hpp"

#include <iostream>
#include <iomanip>
#include <corecrt_math_defines.h>
#include <vector>
#include <utility>
#include <format>
#include <fstream>
#include <chrono>
#include <bitset>
#include <thread>
#include <mutex>
#include <deque>
#include <bit>
#include <random>

#include <Windows.h>
#include <malloc.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#undef min
#undef max
#undef small

#include <mkl.h>
#include <mkl_vml.h>

#define MKL_ILP64
#pragma comment(lib, "mkl_intel_ilp64.lib")
#pragma comment(lib, "mkl_core.lib")
#pragma comment(lib, "mkl_sequential.lib")

struct interval_bit_pattern_iteration_invoker
{
	interval_bit_pattern_iteration_invoker(std::size_t iteration_batch_size, float begin, float end)
		: batch_size_(iteration_batch_size)
	{
		if (batch_size_ == 0) throw std::invalid_argument("iteration_batch_size must be > 0");
		if (std::isnan(begin) || std::isnan(end)) throw std::invalid_argument("begin/end must not be NaN");
		if (begin > end) throw std::invalid_argument("begin must be <= end");

		current_ = float_to_ordered_uint(begin);
		end_ = float_to_ordered_uint(end);

		buffer_.resize(batch_size_);
	}

	template<typename Expr>
		requires std::same_as<std::invoke_result_t<Expr, const float*, std::size_t>, void>
	bool next_batch(Expr&& expr)
	{
		if (finished())
		{
			return false;
		}

		std::uint64_t remaining = static_cast<std::uint64_t>(end_) - static_cast<std::uint64_t>(current_) + 1ull;

		std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(batch_size_, remaining));

		for (std::size_t i = 0; i < n; ++i)
		{
			buffer_[i] = ordered_uint_to_float(current_ + static_cast<std::uint32_t>(i));
		}

		expr(buffer_.data(), n);

		current_ += static_cast<std::uint32_t>(n);
		return true;
	}

	bool finished() const
	{
		return current_ > end_;
	}

	std::uint64_t remaining_count() const
	{
		if (finished())
			return 0;
		return static_cast<std::uint64_t>(end_) - static_cast<std::uint64_t>(current_) + 1ull;
	}

private:
	static std::uint32_t float_to_ordered_uint(float f)
	{
		std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
		return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
	}

	static float ordered_uint_to_float(std::uint32_t u)
	{
		std::uint32_t bits = (u & 0x80000000u) ? (u ^ 0x80000000u) : ~u;
		return std::bit_cast<float>(bits);
	}

	std::size_t batch_size_{};
	std::uint32_t current_{};
	std::uint32_t end_{};
	std::vector<float> buffer_;
};

template<std::floating_point element_type>
struct interval_test_1in_1out_invoker
{
	struct test_result
	{
		double speed_radio;
		std::size_t ULP;
		double reference_cycles_per_call;
		double test_cycles_per_call;
	};

	template<
		typename reference_expr,
		typename test_expr,
		typename progress_expr = std::nullptr_t,
		std::size_t warm_up = 8>
	test_result run(reference_expr&& reference, test_expr&& test,
		element_type minValue, element_type maxValue,
		std::size_t loop_count = 32,
		std::size_t inner_repeat = 64,
		progress_expr&& progress_cb = nullptr)
	{
		const std::uint64_t total_work =
			static_cast<std::uint64_t>(length_) +
			static_cast<std::uint64_t>(warm_up) * 2ull * static_cast<std::uint64_t>(length_) +
			static_cast<std::uint64_t>(loop_count) * static_cast<std::uint64_t>(inner_repeat) * static_cast<std::uint64_t>(length_) +
			static_cast<std::uint64_t>(loop_count) * static_cast<std::uint64_t>(inner_repeat) * static_cast<std::uint64_t>(length_) +
			static_cast<std::uint64_t>(length_);

		std::uint64_t done_work = 0;

		auto report = [&](const char* phase, std::uint64_t inc = 0ull)
			{
				done_work += inc;
				if constexpr (!std::is_same_v<std::decay_t<progress_expr>, std::nullptr_t>)
				{
					progress_cb(done_work, total_work, phase);
				}
			};

		random_fill(minValue, maxValue);
		report("random_fill", static_cast<std::uint64_t>(length_));

		for (std::size_t loop = 0; loop < warm_up; ++loop)
		{
			reference(length_, test_data_, res_0_buffer_);
			report("warmup", static_cast<std::uint64_t>(length_));

			reference(length_, test_data_, res_1_buffer_);
			report("warmup", static_cast<std::uint64_t>(length_));
		}

		std::memset(res_0_buffer_, 0, sizeof(element_type) * length_);
		std::memset(res_1_buffer_, 0, sizeof(element_type) * length_);

		std::uint64_t ref_cycles = 0;
		std::uint64_t test_cycles = 0;

		timeBeginPeriod(1);
		{
			invalidate_buffers();
			for (std::size_t loop = 0; loop < loop_count; ++loop)
			{
				const std::uint64_t begin = tsc_start();

				for (std::size_t r = 0; r < inner_repeat; ++r)
				{
					reference(length_, test_data_, res_0_buffer_);
				}

				const std::uint64_t end = tsc_stop();
				ref_cycles += (end - begin);

				report("reference", 
					static_cast<std::uint64_t>(inner_repeat) * static_cast<std::uint64_t>(length_));
			}

			invalidate_buffers();
			for (std::size_t loop = 0; loop < loop_count; ++loop)
			{
				const std::uint64_t begin = tsc_start();

				for (std::size_t r = 0; r < inner_repeat; ++r)
				{
					test(length_, test_data_, res_1_buffer_);
				}

				const std::uint64_t end = tsc_stop();
				test_cycles += (end - begin);

				report("test", 
					static_cast<std::uint64_t>(inner_repeat) * static_cast<std::uint64_t>(length_));
			}
		}
		timeEndPeriod(1);

		std::size_t max_ulp{ 0 };
		for (std::size_t i = 0; i < length_; ++i)
		{
			max_ulp = std::max(max_ulp,
				acquired_ulp(res_0_buffer_[i], res_1_buffer_[i]));
		}
		report("ulp", static_cast<std::uint64_t>(length_));

		const double total_calls =
			static_cast<double>(loop_count) * static_cast<double>(inner_repeat);

		const double ref_cycles_per_call =
			static_cast<double>(ref_cycles) / total_calls;

		const double test_cycles_per_call =
			static_cast<double>(test_cycles) / total_calls;

		test_result result{};
		result.speed_radio = ref_cycles_per_call / test_cycles_per_call;
		result.ULP = max_ulp;
		result.reference_cycles_per_call = ref_cycles_per_call;
		result.test_cycles_per_call = test_cycles_per_call;

		if constexpr (!std::is_same_v<std::decay_t<progress_expr>, std::nullptr_t>)
		{
			progress_cb(total_work, total_work, "done");
		}

		return result;
	}

	static inline std::uint64_t tsc_start() noexcept
	{
		_mm_lfence();
		return __rdtsc();
	}

	static inline std::uint64_t tsc_stop() noexcept
	{
		unsigned aux = 0;
		const std::uint64_t t = __rdtscp(&aux);
		_mm_lfence();
		return t;
	}

	static void flush_cache_lines(const void* ptr, std::size_t bytes)
	{
		const char* p = static_cast<const char*>(ptr);
		for (std::size_t i = 0; i < bytes; i += 64)
		{
			_mm_clflush(p + i);
		}
	}

	void invalidate_buffers()
	{
		const std::size_t bytes = sizeof(element_type) * length_;
		flush_cache_lines(test_data_, bytes);
		flush_cache_lines(res_0_buffer_, bytes);
		flush_cache_lines(res_1_buffer_, bytes);
		_mm_mfence();
	}

	interval_test_1in_1out_invoker(std::size_t length) : length_(length),
		test_data_(nullptr), res_0_buffer_(nullptr), res_1_buffer_(nullptr)
	{
		test_data_ = reinterpret_cast<element_type*>(_aligned_malloc(sizeof(element_type) * length, 32));
		res_0_buffer_ = reinterpret_cast<element_type*>(_aligned_malloc(sizeof(element_type) * length, 32));
		res_1_buffer_ = reinterpret_cast<element_type*>(_aligned_malloc(sizeof(element_type) * length, 32));

		std::memset(test_data_, 0, sizeof(element_type) * length);
		std::memset(res_0_buffer_, 0, sizeof(element_type) * length);
		std::memset(res_1_buffer_, 0, sizeof(element_type) * length);
	}

	~interval_test_1in_1out_invoker()
	{
		if (test_data_) _aligned_free(test_data_);
		if (res_0_buffer_) _aligned_free(res_0_buffer_);
		if (res_1_buffer_) _aligned_free(res_1_buffer_);
	}

	std::uint32_t float_to_ordered_uint(float x)
	{
		std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
		return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
	}

	float ordered_uint_to_float(std::uint32_t u)
	{
		std::uint32_t bits = (u & 0x80000000u) ? (u ^ 0x80000000u) : ~u;
		return std::bit_cast<float>(bits);
	}

	void random_fill(element_type minValue, element_type maxValue)
	{
		static_assert(std::is_same_v<element_type, float>, "This implementation is for float only");
		static_assert(sizeof(element_type) == sizeof(std::uint32_t), "element_type must be 32-bit float");

		static thread_local std::mt19937_64 gen{ std::random_device{}() };

		std::uint32_t lo = float_to_ordered_uint(minValue);
		std::uint32_t hi = float_to_ordered_uint(maxValue);

		std::uniform_int_distribution<std::uint32_t> dist(lo, hi);

		for (std::size_t i = 0; i < length_; ++i)
		{
			std::uint32_t u = dist(gen);
			test_data_[i] = ordered_uint_to_float(u);
		}
	}
	
	void empty_all()
	{
		std::memset(res_0_buffer_, 0, sizeof(element_type) * length_);
		std::memset(res_1_buffer_, 0, sizeof(element_type) * length_);
		std::memset(test_data_, 0, sizeof(element_type) * length_);
	}

	static constexpr std::size_t acquired_ulp(element_type a, element_type b) noexcept
	{
		using UInt = std::conditional_t<std::is_same_v<element_type, float>, std::uint32_t, std::uint64_t>;

		constexpr UInt sign_mask = [] {
			if constexpr (std::is_same_v<element_type, float>) { return UInt{ 0x80000000u }; }
			else { return UInt{ 0x8000000000000000ull }; }
			}();

		const UInt ua = std::bit_cast<UInt>(a);
		const UInt ub = std::bit_cast<UInt>(b);

		const bool sign_a = (ua & sign_mask) != 0;
		const bool sign_b = (ub & sign_mask) != 0;

		const bool a_finite = std::isfinite(a);
		const bool b_finite = std::isfinite(b);

		if (!a_finite || !b_finite)
		{
			if (a_finite != b_finite)
			{
				return std::numeric_limits<UInt>::max();
			}

			if (sign_a != sign_b)
			{
				return std::numeric_limits<UInt>::max();
			}

			return (ua == ub) ? UInt{ 0 } : std::numeric_limits<UInt>::max();
		}

		if (a == element_type{ 0 } && b == element_type{ 0 } && sign_a != sign_b)
		{
			return std::numeric_limits<UInt>::max();
		}

		if (ua == ub)
		{
			return UInt{ 0 };
		}

		const auto toOrdered = [sign_mask](UInt bits) constexpr noexcept -> UInt
			{
				return (bits & sign_mask) ? ~bits : (bits | sign_mask);
			};

		const UInt oa = toOrdered(ua);
		const UInt ob = toOrdered(ub);

		return (oa >= ob) ? (oa - ob) : (ob - oa);
	}

	std::size_t length_;
	element_type* test_data_;
	element_type* res_0_buffer_;
	element_type* res_1_buffer_;
};

static std::vector<DWORD_PTR> enumerate_physical_core_masks()
{
	std::vector<DWORD_PTR> core_masks;

	DWORD len = 0;
	GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
	if (len == 0)
	{
		return core_masks;
	}

	std::vector<std::byte> buffer(len);
	auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());

	if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len))
	{
		return core_masks;
	}

	const std::byte* ptr = buffer.data();
	const std::byte* end = buffer.data() + len;

	while (ptr < end)
	{
		auto* entry = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
			const_cast<std::byte*>(ptr));

		if (entry->Relationship == RelationProcessorCore)
		{
			for (WORD g = 0; g < entry->Processor.GroupCount; ++g)
			{
				const GROUP_AFFINITY& ga = entry->Processor.GroupMask[g];

				if (ga.Group == 0 && ga.Mask != 0)
				{
					DWORD_PTR first_logical = ga.Mask & (~ga.Mask + 1);
					core_masks.push_back(first_logical);
				}
			}
		}

		ptr += entry->Size;
	}

	return core_masks;
}

static bool bind_current_thread_to_physical_core(
	const std::vector<DWORD_PTR>& physical_core_masks,
	std::size_t core_index)
{
	if (physical_core_masks.empty())
	{
		return false;
	}

	const DWORD_PTR mask =
		physical_core_masks[core_index % physical_core_masks.size()];

	return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

template<std::floating_point element_type,
	std::size_t length, std::size_t loop,
	typename reference_expr, typename test_expr>
void run_1in_1out_test(
	const std::vector<std::tuple<std::string, float, float>>& test_range,
	reference_expr&& reference, test_expr&& test)
{
	using invoker_t = interval_test_1in_1out_invoker<element_type>;
	using result_t = typename invoker_t::test_result;

	struct case_result
	{
		std::string name;
		element_type range_min;
		element_type range_max;
		result_t result;
	};

	enum class case_state : std::uint8_t
	{
		waiting,
		running,
		done
	};

	enum class case_phase : std::uint8_t
	{
		none,
		random_fill,
		warmup,
		reference,
		test,
		ulp,
		done
	};

	struct case_progress
	{
		std::atomic<case_state> state;
		std::atomic<int> worker_id;
		std::atomic<std::uint64_t> processed;
		std::atomic<std::uint64_t> total;
		std::atomic<case_phase> phase;

		case_progress()
			: state(case_state::waiting),
			worker_id(-1),
			processed(0),
			total(0),
			phase(case_phase::none)
		{
		}

		case_progress(const case_progress&) = delete;
		case_progress& operator=(const case_progress&) = delete;
	};

	const std::size_t total_cases = test_range.size();
	if (total_cases == 0)
	{
		std::cout << "no test cases\n";
		return;
	}

	const auto physical_core_masks = enumerate_physical_core_masks();

	std::size_t usable_threads = 0;
	if (!physical_core_masks.empty())
	{
		usable_threads = physical_core_masks.size();
	}
	else
	{
		unsigned hw_threads = std::thread::hardware_concurrency();
		if (hw_threads == 0)
		{
			hw_threads = 4;
		}
		usable_threads = static_cast<std::size_t>(hw_threads);
	}

	const std::size_t thread_count =
		std::min<std::size_t>(usable_threads, total_cases);

	std::vector<std::optional<case_result>> all_results(total_cases);
	auto progresses = std::make_unique<case_progress[]>(total_cases);

	std::vector<std::thread> workers;
	workers.reserve(thread_count);

	std::atomic<std::size_t> next_index{ 0 };
	std::atomic<std::size_t> finished_count{ 0 };

	auto to_phase_enum = [](const char* s) -> case_phase
		{
			if (std::strcmp(s, "random_fill") == 0) return case_phase::random_fill;
			if (std::strcmp(s, "warmup") == 0)      return case_phase::warmup;
			if (std::strcmp(s, "reference") == 0)   return case_phase::reference;
			if (std::strcmp(s, "test") == 0)        return case_phase::test;
			if (std::strcmp(s, "ulp") == 0)         return case_phase::ulp;
			if (std::strcmp(s, "done") == 0)        return case_phase::done;
			return case_phase::none;
		};

	auto phase_to_string = [](case_phase p) -> const char*
		{
			switch (p)
			{
				case case_phase::random_fill: return "random_fill";
				case case_phase::warmup:      return "warmup";
				case case_phase::reference:   return "reference";
				case case_phase::test:        return "test";
				case case_phase::ulp:         return "ulp";
				case case_phase::done:        return "done";
				default:                      return "";
			}
		};

	for (std::size_t worker_idx = 0; worker_idx < thread_count; ++worker_idx)
	{
		workers.emplace_back([&, worker_idx]()
		{
			if (!physical_core_masks.empty())
			{
				bind_current_thread_to_physical_core(physical_core_masks, worker_idx);
			}

			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

			for (;;)
			{
				const std::size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
				if (idx >= total_cases)
				{
					break;
				}

				progresses[idx].processed.store(0, std::memory_order_release);
				progresses[idx].total.store(0, std::memory_order_release);
				progresses[idx].phase.store(case_phase::none, std::memory_order_release);
				progresses[idx].worker_id.store(static_cast<int>(worker_idx), std::memory_order_release);
				progresses[idx].state.store(case_state::running, std::memory_order_release);

				const std::string& name = std::get<0>(test_range[idx]);
				const element_type range_min = static_cast<element_type>(std::get<1>(test_range[idx]));
				const element_type range_max = static_cast<element_type>(std::get<2>(test_range[idx]));

				invoker_t invoker(length);

				result_t result = invoker.run(
					[&](std::size_t n, const element_type* in, element_type* out) -> void { reference(n, in, out); },
					[&](std::size_t n, const element_type* in, element_type* out) -> void { test(n, in, out); },
					range_min,
					range_max,
					loop,
					8,
					[&](std::uint64_t done_work, std::uint64_t total_work, const char* phase_name)
					{
						progresses[idx].processed.store(done_work, std::memory_order_release);
						progresses[idx].total.store(total_work, std::memory_order_release);
						progresses[idx].phase.store(to_phase_enum(phase_name), std::memory_order_release);
					}
				);

				all_results[idx].emplace(case_result{ name, range_min, range_max, std::move(result) });

				progresses[idx].state.store(case_state::done, std::memory_order_release);
				progresses[idx].phase.store(case_phase::done, std::memory_order_release);
				finished_count.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}

	constexpr int col_idx_width = 8;
	constexpr int col_name_width = 24;
	constexpr int col_range_width = 34;
	constexpr int col_status_width = 48;

	const std::size_t line_count = total_cases + 2;
	std::cout << "\x1b[2J\x1b[H";
	for (std::size_t i = 0; i < line_count; ++i)
	{
		std::cout << '\n';
	}
	std::cout << "\x1b[H";

	while (true)
	{
		const std::size_t done = finished_count.load(std::memory_order_acquire);

		std::cout << "\x1b[H";
		std::cout << std::format(
			"{:<12}{:<16}{:<18}\n",
			"cases",
			std::format("{}/{}", done, total_cases),
			std::format("threads {}", thread_count)
		);

		std::cout << std::format(
			"{:<{}}{:<{}}{:<{}}{:<{}}\n",
			"index", col_idx_width,
			"name", col_name_width,
			"range", col_range_width,
			"     status", col_status_width
		);

		for (std::size_t idx = 0; idx < total_cases; ++idx)
		{
			const auto state = progresses[idx].state.load(std::memory_order_acquire);
			const int worker_id = progresses[idx].worker_id.load(std::memory_order_acquire);
			const std::uint64_t processed = progresses[idx].processed.load(std::memory_order_acquire);
			const std::uint64_t total = progresses[idx].total.load(std::memory_order_acquire);
			const case_phase phase = progresses[idx].phase.load(std::memory_order_acquire);

			const std::string& name = std::get<0>(test_range[idx]);
			const element_type range_min = static_cast<element_type>(std::get<1>(test_range[idx]));
			const element_type range_max = static_cast<element_type>(std::get<2>(test_range[idx]));

			const std::string idx_text = std::format("[{}]", idx);
			const std::string range_text = std::format("[{:.8e}, {:.8e}]", range_min, range_max);

			std::string status_text;

			switch (state)
			{
			case case_state::waiting:
				status_text = "waiting";
				break;

			case case_state::running:
			{
				const double percent = (total == 0)
					? 0.0
					: 100.0 * static_cast<double>(std::min(processed, total))
					/ static_cast<double>(total);

				status_text = std::format(
					"running {:>6.2f}% {:<10} (worker {})",
					percent,
					phase_to_string(phase),
					worker_id
				);
				break;
			}

			case case_state::done:
			{
				const auto& r = *all_results[idx];
				status_text = std::format(
					"radio {:>.6f} x  max ulp {}",
					r.result.speed_radio,
					r.result.ULP
				);
				break;
			}
			}

			std::cout << std::format(
				"{:<{}}{:<{}}{:<{}}{:<{}}\n",
				idx_text, col_idx_width,
				name, col_name_width,
				range_text, col_range_width,
				std::format("     {}", status_text), col_status_width
			);
		}

		std::cout << std::flush;

		if (done == total_cases)
		{
			break;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}

	for (auto& t : workers)
	{
		t.join();
	}

	std::cout << "\nfinished.\n";
}

static std::vector<std::tuple<std::string, float, float>> expm1_fp32_test_range = {
	{"scale 1e-1",					 -1e-1f,       1e-1f},
	{"scale 1e-2",					 -1e-2f,       1e-2f},
	{"small poly",					 -1e-3f,       1e-3f},
	{"scale 1e-4",					 -1e-4f,       1e-4f},
	{"scale 1e-6",					 -1e-6f,       1e-6f},
	{"tiny region",					 -1e-8f,       1e-8f},
	{"tiny deep",					-1e-12f,      1e-12f},
	{"half ln2 inside",			-0.3465735f,  0.3465735f},
	{"k pm1 pos",				 0.3465737f,  1.0397207f},
	{"k pm1 neg",				-1.0397207f, -0.3465737f},
	{"general pos",				   1.039721f,  18.71497f},
	{"general neg",				-18.71497f,   -1.039721f},
	{"huge neg to minus1",		 -100.0f,      -18.7151f},
	{"large pos finite",			  18.7151f,    80.0f},
	{"overflow window",				  88.0f,       89.0f},
	{"tiny fast return",		   -2.9e-8f,     2.9e-8f},
	{"large pos to ovf",		   18.7151f,    88.7228f},
	{"overflow boundary",			 88.72f,      88.73f},
	{"overflow pos",				  88.723f,    100.0f},
};

static std::vector<std::tuple<std::string, float, float>> exp_fp32_test_range = {
		{"tiny around 0",				  -1e-06, 1e-06},
		{"small around 0",				  -0.001, 0.001},
		{"probability scale",				  -0.1, 0.1},
		{"unit scale",							  -1, 1},
		{"ml/logit common",						-10, 10},
		{"wide common",							-40, 40},
		{"half ln2 inside",				-0.3465, 0.3465},
		{"k around +1",					 0.3466, 1.0397},
		{"k around -1",				   -1.0397, -0.3466},
		{"general positive",			     1.0398, 10},
		{"general negative",		       -10, -1.0398},
		{"half ln2 boundary +",          0.3464, 0.3468},
		{"half ln2 boundary -",        -0.3468, -0.3464},
		{"1.5 ln2 boundary +",             1.0395, 1.04},
		{"1.5 ln2 boundary -",           -1.04, -1.0395},
		{"normal low boundary",              -88, -87.2},
		{"subnormal region",                -103, -87.4},
		{"underflow boundary",           -104.1, -103.7},
		{"hard underflow",			       -150, -104.1},
		{"high finite normal",			         80, 88},
		{"k128 neighborhood",			      88, 88.72},
		{"overflow boundary",			   88.72, 88.73},
		{"overflow positive",			    88.723, 100},
};

static std::vector<std::tuple<std::string, float, float>> exp2_fp32_test_range = {
        {"tiny core",              -1e-8f,     1e-8f},
        {"tiny threshold mix",     -4e-8f,     4e-8f},
        {"minus0p5 0p5",           -0.5f,       0.5f},
        {"minus1 1",               -1.0f,       1.0f},
        {"minus8 8",               -8.0f,       8.0f},
        {"minus32 32",            -32.0f,      32.0f},
        {"minus100 100",         -100.0f,     100.0f},
        {"full fast domain",     -126.0f,  127.9375f},
        {"norm subnorm edge",    -127.0f,    -125.0f},
        {"light subnormal",      -130.0f,    -126.0f},
        {"deep subnormal",       -149.0f,    -130.0f},
        {"underflow edge",       -151.0f,    -149.0f},
        {"pure underflow",       -200.0f,    -160.0f},
        {"overflow edge",         127.0f,     129.0f},
        {"full mixed",           -200.0f,     150.0f},
        {"near zero common",       -0.125f,   0.125f},
        {"small common",           -2.0f,       2.0f},
        {"mainstream common",      -5.0f,       5.0f},
        {"typical common",        -10.0f,      10.0f},
        {"medium common",         -16.0f,      16.0f},
        {"decay common",          -20.0f,       0.0f},
        {"strong decay common",   -40.0f,       0.0f},
        {"wide realistic",        -40.0f,      10.0f},
};

std::vector<std::tuple<std::string, float, float>> exp10_fp32_test_range = {
        {"tiny core",             -1e-8f,    1e-8f},
        {"tiny threshold mix",    -4e-8f,    4e-8f},
        {"near zero small",       -0.1f,      0.1f},
        {"minus1 1",              -1.0f,      1.0f},
        {"minus2 2",              -2.0f,      2.0f},
        {"minus5 5",              -5.0f,      5.0f},
        {"minus10 10",           -10.0f,     10.0f},
		{"pos0 10",                 0.0,     10.0f},
		{"ne20 ne10",            -20.0f,    -10.0f},
        {"decay common",         -20.0f,      0.0f},
        {"wide realistic",       -20.0f,     10.0f},
        {"norm subnorm_edge",    -38.5f,    -37.0f},
        {"light subnormal",      -40.0f,    -38.0f},
        {"deep subnormal",       -44.5f,    -40.0f},
        {"underflow edge",       -45.5f,    -44.0f},
        {"pure underflow",       -60.0f,    -50.0f},
        {"overflow edge",         38.0f,     39.0f},
        {"full mixed",           -60.0f,     40.0f},
};

std::vector<std::tuple<std::string, float, float>> ln_fp32_test_range = {
	{"near1_ultra_tight",      0.9999990f,        1.0000010f},
	{"near1_very_tight",       0.9999000f,        1.0001000f},
	{"near1_tight",            0.9990000f,        1.0010000f},
	{"near1_left_half",        0.7500010f,        0.9999990f},
	{"near1_right_half",       1.0000010f,        1.2499990f},
	{"near1_full_window",      0.7500010f,        1.2499990f},
	{"below_near1_threshold",  0.5000000f,        0.7499000f},
	{"above_near1_threshold",  1.2501000f,        2.0000000f},
	{"near1_boundary_mix",     0.7400000f,        1.2600000f},
	{"positive_small_unit",    0.1000000f,        1.0000000f},
	{"positive_unit_to_10",    1.0000000f,        10.0000000f},
	{"positive_0p25_to_4",     0.2500000f,        4.0000000f},
	{"positive_0p5_to_2",      0.5000000f,        2.0000000f},
	{"pow2_mid_span",          9.7656250e-4f,     1.0240000e+3f},
	{"pow2_wide_span",         9.5367432e-7f,     1.0485760e+6f},
	{"tiny_normal_band",       1.1754944e-38f,    1.0000000e-30f},
	{"subnormal_only",         1.4012985e-45f,    1.1754942e-38f},
	{"subnormal_and_normal",   1.4012985e-45f,    1.0000000e-37f},
	{"large_positive",         1.0000000e+3f,     1.0000000e+10f},
	{"huge_positive",          1.0000000e+10f,    3.4028235e+38f},
	{"negative_small",        -1.0000000e-6f,    -1.0000000e-12f},
	{"negative_unit",         -10.0000000f,       -0.1000000f},
	{"mixed_sign_small",      -1.0000000e-3f,      1.0000000e-3f},
	{"mixed_sign_unit",       -1.0000000f,         1.0000000f}
};

std::vector<std::tuple<std::string, float, float>> log2_fp32_test_range = {
	{"near_1_ultra_tight",     0.9999990f,        1.0000010f},
	{"near_1_very_tight",      0.9999000f,        1.0001000f},
	{"near_1_tight",           0.9990000f,        1.0010000f},
	{"near_0p5",               0.4995000f,        0.5005000f},
	{"near_2",                 1.9990000f,        2.0010000f},
	{"near_4",                 3.9980000f,        4.0020000f},
	{"near_1024",              1023.0000f,        1025.0000f},
	{"near_1_over_1024",       9.7550000e-4f,     9.7750000e-4f},
	{"positive_0p5_to_2",      0.5000000f,        2.0000000f},
	{"positive_0p25_to_4",     0.2500000f,        4.0000000f},
	{"positive_1_to_10",       1.0000000f,        10.0000000f},
	{"positive_0p1_to_10",     0.1000000f,        10.0000000f},
	{"pow2_mid_span",          9.7656250e-4f,     1.0240000e+3f},
	{"pow2_wide_span",         9.5367432e-7f,     1.0485760e+6f},
	{"wide_positive",          1.0000000e-20f,    1.0000000e+20f},
	{"tiny_normal_band",       1.1754944e-38f,    1.0000000e-30f},
	{"subnormal_only",         1.4012985e-45f,    1.1754942e-38f},
	{"subnormal_and_normal",   1.4012985e-45f,    1.0000000e-37f},
	{"large_positive",         1.0000000e+3f,     1.0000000e+10f},
	{"huge_positive",          1.0000000e+10f,    3.4028235e+38f},
	{"zero_to_tiny",           0.0000000f,        1.0000000e-37f},
	{"negative_small",        -1.0000000e-6f,    -1.0000000e-12f},
	{"negative_unit",         -10.0000000f,       -0.1000000f},
	{"mixed_sign_small",      -1.0000000e-3f,      1.0000000e-3f},
	{"mixed_sign_unit",       -1.0000000f,         1.0000000f}
};

std::vector<std::tuple<std::string, float, float>> log10_fp32_test_range = {
	{"near_1_ultra_tight",      0.9999990f,        1.0000010f},
	{"near_1_very_tight",       0.9999000f,        1.0001000f},
	{"near_1_tight",            0.9990000f,        1.0010000f},
	{"near_0p5",                0.4995000f,        0.5005000f},
	{"near_2",                  1.9990000f,        2.0010000f},
	{"near_10",                 9.9900000f,       10.0100000f},
	{"near_100",               99.9000000f,      100.1000000f},
	{"near_0p1",                0.0999000f,        0.1001000f},
	{"positive_0p1_to_1",       0.1000000f,        1.0000000f},
	{"positive_1_to_10",        1.0000000f,       10.0000000f},
	{"positive_0p1_to_10",      0.1000000f,       10.0000000f},
	{"positive_0p01_to_100",    0.0100000f,      100.0000000f},
	{"positive_0p5_to_2",       0.5000000f,        2.0000000f},
	{"pow10_mid_span",          1.0000000e-5f,     1.0000000e+5f},
	{"pow10_wide_span",         1.0000000e-20f,    1.0000000e+20f},
	{"tiny_normal_band",        1.1754944e-38f,    1.0000000e-30f},
	{"subnormal_only",          1.4012985e-45f,    1.1754942e-38f},
	{"subnormal_and_normal",    1.4012985e-45f,    1.0000000e-37f},
	{"large_positive",          1.0000000e+3f,     1.0000000e+10f},
	{"huge_positive",           1.0000000e+10f,    3.4028235e+38f},
	{"zero_to_tiny",            0.0000000f,        1.0000000e-37f},
	{"negative_small",         -1.0000000e-6f,    -1.0000000e-12f},
	{"negative_unit",          -10.0000000f,       -0.1000000f},
	{"mixed_sign_small",       -1.0000000e-3f,      1.0000000e-3f},
	{"mixed_sign_unit",        -1.0000000f,         1.0000000f}
};

std::vector<std::tuple<std::string, float, float>> log1p_fp32_test_range = {
	{"tiny_symmetric_ultra",    -1.0e-8f,          1.0e-8f},
	{"tiny_symmetric_very",     -1.0e-6f,          1.0e-6f},
	{"tiny_symmetric",          -1.0e-4f,          1.0e-4f},
	{"small_positive",           0.0f,             1.0e-3f},
	{"small_positive_wide",      0.0f,             1.0e-1f},
	{"small_negative",          -1.0e-3f,          0.0f},
	{"small_negative_wide",     -1.0e-1f,          0.0f},
	{"small_symmetric",         -1.0e-2f,          1.0e-2f},
	{"medium_symmetric",        -1.0e-1f,          1.0e-1f},
	{"near_minus1_ultra",       -0.9999990f,      -0.9999900f},
	{"near_minus1_very",        -0.9999000f,      -0.9990000f},
	{"near_minus1",             -0.9990000f,      -0.9900000f},
	{"approach_minus1_mixed",   -0.9999990f,      -0.9000000f},
	{"negative_valid_mid",      -0.9000000f,      -0.1000000f},
	{"negative_valid_full",     -0.9990000f,       0.0000000f},
	{"positive_unit",            0.0000000f,       1.0000000f},
	{"positive_0_to_10",         0.0000000f,      10.0000000f},
	{"positive_1_to_100",        1.0000000f,     100.0000000f},
	{"large_positive",           1.0000000e+3f,    1.0000000e+10f},
	{"huge_positive",            1.0000000e+10f,   3.4028235e+38f},
	{"invalid_below_minus1_small", -1.0010000f,   -1.0000010f},
	{"invalid_below_minus1_mid",   -2.0000000f,   -1.0010000f},
	{"invalid_below_minus1_wide", -100.0000000f,  -1.0001000f},
	{"cross_minus1_boundary",   -1.1000000f,      -0.9000000f},
	{"mixed_sign_with_boundary",-1.1000000f,       1.0000000f}
};

std::vector<std::tuple<std::string, float, float>> logb_fp32_test_range = {
	{"near_1_ultra_tight",      0.9999990f,        1.0000010f},
	{"near_1_tight",            0.9990000f,        1.0010000f},
	{"same_exp_0p5_to_1",       0.5000000f,        0.9999990f},
	{"same_exp_1_to_2",         1.0000000f,        1.9999990f},
	{"same_exp_2_to_4",         2.0000000f,        3.9999990f},
	{"same_exp_1024_to_2048",   1024.0000f,     2047.9990f},
	{"positive_0p1_to_10",      0.1000000f,       10.0000000f},
	{"positive_0p01_to_100",    0.0100000f,      100.0000000f},
	{"pow2_mid_span",           9.7656250e-4f,     1.0240000e+3f},
	{"pow2_wide_span",          9.5367432e-7f,     1.0485760e+6f},
	{"wide_positive",           1.0000000e-20f,    1.0000000e+20f},
	{"tiny_normal_band",        1.1754944e-38f,    1.0000000e-30f},
	{"subnormal_only",          1.4012985e-45f,    1.1754942e-38f},
	{"subnormal_and_normal",    1.4012985e-45f,    1.0000000e-37f},
	{"large_positive",          1.0000000e+3f,     1.0000000e+10f},
	{"huge_positive",           1.0000000e+10f,    3.4028235e+38f},
	{"negative_same_exp",      -2.0000000f,       -1.0000000f},
	{"negative_wide",          -1.0000000e+10f,   -1.0000000e-10f},
	{"zero_to_tiny",            0.0000000f,        1.0000000e-37f},
	{"mixed_zero_small",       -1.0000000e-6f,     1.0000000e-6f},
	{"mixed_sign_unit",        -1.0000000f,         1.0000000f},
	{"mixed_sign_wide",        -1.0000000e+6f,      1.0000000e+6f}
};

std::vector<std::tuple<std::string, float, float>> cbrt_fp32_test_range = {
	{"near_1_ultra_tight",       0.9999990f,        1.0000010f},
	{"near_1_tight",             0.9990000f,        1.0010000f},
	{"near_1_wide",              0.9000000f,        1.1000000f},
	{"positive_0p1_to_10",       0.1000000f,       10.0000000f},
	{"positive_1_to_100",        1.0000000f,      100.0000000f},
	{"positive_0p01_to_100",     0.0100000f,      100.0000000f},
	{"mixed_sign_small",        -1.0000000e-3f,     1.0000000e-3f},
	{"mixed_sign_unit",         -1.0000000f,        1.0000000f},
	{"mixed_sign_medium",      -100.0000000f,     100.0000000f},
	{"negative_small",          -1.0000000e-3f,    -1.0000000e-6f},
	{"negative_unit",          -10.0000000f,       -0.1000000f},
	{"negative_large",         -1.0000000e+6f,     -1.0000000f},
	{"tiny_normal_band",         1.1754944e-38f,    1.0000000e-30f},
	{"subnormal_only",           1.4012985e-45f,    1.1754942e-38f},
	{"subnormal_and_normal",     1.4012985e-45f,    1.0000000e-37f},
	{"negative_tiny_normal",    -1.0000000e-30f,   -1.1754944e-38f},
	{"negative_subnormal_only", -1.1754942e-38f,   -1.4012985e-45f},
	{"large_positive",           1.0000000e+3f,     1.0000000e+10f},
	{"huge_positive",            1.0000000e+10f,    3.4028235e+38f},
	{"large_negative",          -1.0000000e+10f,   -1.0000000e+3f},
	{"wide_positive",            1.0000000e-20f,    1.0000000e+20f},
	{"wide_mixed_sign",         -1.0000000e+20f,    1.0000000e+20f}
};

std::vector<std::tuple<std::string, float, float>> invcbrt_fp32_test_range = {
	{"near_1_ultra_tight",       0.9999990f,        1.0000010f},
	{"near_1_tight",             0.9990000f,        1.0010000f},
	{"near_1_wide",              0.9000000f,        1.1000000f},
	{"positive_0p1_to_10",       0.1000000f,       10.0000000f},
	{"positive_1_to_100",        1.0000000f,      100.0000000f},
	{"positive_0p01_to_100",     0.0100000f,      100.0000000f},
	{"positive_near_zero_tiny",  1.0000000e-12f,    1.0000000e-6f},
	{"positive_near_zero_small", 1.0000000e-6f,     1.0000000e-3f},
	{"positive_near_zero_wide",  1.0000000e-20f,    1.0000000e-3f},
	{"negative_small",          -1.0000000e-3f,    -1.0000000e-6f},
	{"negative_unit",          -10.0000000f,       -0.1000000f},
	{"negative_large",         -1.0000000e+6f,     -1.0000000f},
	{"negative_near_zero_tiny", -1.0000000e-6f,    -1.0000000e-12f},
	{"negative_near_zero_wide", -1.0000000e-3f,    -1.0000000e-20f},
	{"tiny_normal_band",         1.1754944e-38f,    1.0000000e-30f},
	{"subnormal_only",           1.4012985e-45f,    1.1754942e-38f},
	{"subnormal_and_normal",     1.4012985e-45f,    1.0000000e-37f},
	{"negative_tiny_normal",    -1.0000000e-30f,   -1.1754944e-38f},
	{"negative_subnormal_only", -1.1754942e-38f,   -1.4012985e-45f},
	{"large_positive",           1.0000000e+3f,     1.0000000e+10f},
	{"huge_positive",            1.0000000e+10f,    3.4028235e+38f},
	{"large_negative",          -1.0000000e+10f,   -1.0000000e+3f},
	{"mixed_sign_small",        -1.0000000e-3f,     1.0000000e-3f},
	{"mixed_sign_unit",         -1.0000000f,        1.0000000f},
	{"wide_mixed_sign",         -1.0000000e+20f,    1.0000000e+20f}
};

struct alignas(64) ThreadProgress
{
	std::uint64_t range_begin = 0;
	std::uint64_t range_end = 0;

	std::atomic<std::uint64_t> done_values{ 0 };
	std::atomic<std::size_t> max_ulp{ 0 };
	std::atomic<bool> finished{ false };
};

template<typename before_expr, typename after_expr, 
	typename reference_expr, typename test_expr>
void run_1in_1out_domain_fp32_test(
	before_expr&& before, after_expr&& after,
	reference_expr&& reference_func, test_expr&& test_func)
{
	before();

	constexpr std::uint64_t total_values = (1ull << 32);
	constexpr std::size_t report_block_values = 64ull * 1024ull;

	unsigned hw_threads = std::thread::hardware_concurrency();
	if (hw_threads == 0)
	{
		hw_threads = 4;
	}

	const std::size_t thread_count = static_cast<std::size_t>(hw_threads);
	const std::uint64_t values_per_thread = (total_values + thread_count - 1) / thread_count;

	std::vector<ThreadProgress> progresses(thread_count);
	std::vector<std::thread> workers;
	workers.reserve(thread_count);

	for (std::size_t tid = 0; tid < thread_count; ++tid)
	{
		const std::uint64_t begin = values_per_thread * tid;
		const std::uint64_t end = std::min(begin + values_per_thread, total_values);

		progresses[tid].range_begin = begin;
		progresses[tid].range_end = end;
	}

	for (std::size_t tid = 0; tid < thread_count; ++tid)
	{
		workers.emplace_back([tid, &progresses, &reference_func, &test_func, report_block_values]()
			{
				ThreadProgress& prog = progresses[tid];

				const std::uint64_t begin = prog.range_begin;
				const std::uint64_t end = prog.range_end;

				std::vector<float> current(report_block_values);
				std::vector<double> reference_src(report_block_values);
				std::vector<double> reference_res(report_block_values);
				std::vector<float> test_array(report_block_values);

				std::size_t local_max_ulp = 0;
				std::uint64_t local_done = 0;

				for (std::uint64_t block_begin = begin; block_begin < end; block_begin += report_block_values)
				{
					const std::uint64_t remain = end - block_begin;
					const std::size_t n = static_cast<std::size_t>(
						std::min<std::uint64_t>(report_block_values, remain));

					for (std::size_t i = 0; i < n; ++i)
					{
						const std::uint32_t bits = static_cast<std::uint32_t>(block_begin + i);
						const float x = std::bit_cast<float>(bits);
						current[i] = x;
						reference_src[i] = static_cast<double>(x);
					}

					reference_func(static_cast<int>(n), reference_src.data(), reference_res.data());
					test_func(static_cast<int>(n), current.data(), test_array.data());

					for (std::size_t i = 0; i < n; ++i)
					{
						const float ref_value = static_cast<float>(reference_res[i]);
						const float test_value = test_array[i];

						const std::size_t ulp =
							interval_test_1in_1out_invoker<float>::acquired_ulp(ref_value, test_value);

						if (ulp > local_max_ulp)
						{
							local_max_ulp = ulp;
							prog.max_ulp.store(local_max_ulp, std::memory_order_release);
						}
					}

					local_done += n;
					prog.done_values.store(local_done, std::memory_order_release);
				}

				prog.max_ulp.store(local_max_ulp, std::memory_order_release);
				prog.finished.store(true, std::memory_order_release);
			}
		);
	}

	const std::size_t line_count = thread_count + 1;

	std::cout << "\x1b[2J\x1b[H";
	for (std::size_t i = 0; i < line_count; ++i)
	{
		std::cout << '\n';
	}
	std::cout << "\x1b[H";

	while (true)
	{
		bool all_finished = true;
		std::uint64_t overall_done = 0;
		std::size_t overall_max_ulp = 0;

		for (std::size_t tid = 0; tid < thread_count; ++tid)
		{
			overall_done += progresses[tid].done_values.load(std::memory_order_acquire);
			overall_max_ulp = std::max(
				overall_max_ulp,
				progresses[tid].max_ulp.load(std::memory_order_acquire));

			if (!progresses[tid].finished.load(std::memory_order_acquire))
			{
				all_finished = false;
			}
		}

		std::cout << "\x1b[H";

		{
			const double percent =
				100.0 * static_cast<double>(overall_done) / static_cast<double>(total_values);

			std::cout << std::format(
				"overall  {:>6.2f}%  values: {}/{}  max ulp: {}\n",
				percent,
				overall_done,
				total_values,
				overall_max_ulp
			);
		}

		for (std::size_t tid = 0; tid < thread_count; ++tid)
		{
			const auto& prog = progresses[tid];

			const std::uint64_t begin = prog.range_begin;
			const std::uint64_t end = prog.range_end;
			const std::uint64_t total = end - begin;
			const std::uint64_t done = prog.done_values.load(std::memory_order_acquire);
			const std::size_t maxulp = prog.max_ulp.load(std::memory_order_acquire);
			const bool finished = prog.finished.load(std::memory_order_acquire);

			const double percent = (total == 0)
				? 100.0
				: 100.0 * static_cast<double>(done) / static_cast<double>(total);

			const std::uint32_t begin32 = static_cast<std::uint32_t>(begin);
			const std::uint32_t end32 = static_cast<std::uint32_t>(end - 1);

			std::cout << std::format(
				"thread {:>2}  range:[0x{:08X}, 0x{:08X}]  {:>6.2f}%  values: {}/{}  max ulp: {}{}\n",
				tid,
				begin32,
				end32,
				percent,
				done,
				total,
				maxulp,
				finished ? "  done" : ""
			);
		}

		std::cout << std::flush;

		if (all_finished)
		{
			break;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}

	for (auto& t : workers)
	{
		t.join();
	}

	std::cout << "\nfinished.\n";
	after();
}

std::vector<std::tuple<std::string, float, float>> erf_fp32_test_range = {
	{"small_pos_tight_near_zero",   0.0000000f,   0.0312500f},
	{"small_pos_full",              0.0000000f,   0.8437000f},
	{"small_neg_full",             -0.8437000f,   0.0000000f},
	{"small_mixed_full",           -0.8437000f,   0.8437000f},

	{"boundary_0p84375_tight_pos",  0.8430000f,   0.8445000f},
	{"boundary_0p84375_mixed",     -0.8445000f,   0.8445000f},

	{"mid_pos_full",                0.8437500f,   1.2499000f},
	{"mid_neg_full",               -1.2499000f,  -0.8437500f},
	{"mid_mixed_full",             -1.2499000f,   1.2499000f},

	{"boundary_1p25_tight_pos",     1.2450000f,   1.2550000f},
	{"boundary_1p25_mixed",        -1.2550000f,   1.2550000f},

	{"tail1_pos_full",              1.2500000f,   2.8570000f},
	{"tail1_mixed_full",           -2.8570000f,   2.8570000f},

	{"boundary_2p857143_tight_pos", 2.8500000f,   2.8650000f},
	{"boundary_2p857143_mixed",    -2.8650000f,   2.8650000f},

	{"tail2_pos_full",              2.8571430f,   3.9999000f},
	{"tail2_mixed_full",           -3.9999000f,   3.9999000f},

	{"boundary_4p0_tight_mixed",   -4.0100000f,   4.0100000f},
	{"realistic_centered_unit",    -1.5000000f,   1.5000000f},
	{"realistic_centered_wide",    -4.0000000f,   4.0000000f}
};

std::vector<std::tuple<std::string, float, float>> erfc_fp32_test_range = {
	{"small_pos_tight_near_zero",    0.0000000f,   0.0312500f},
	{"small_pos_micro",              0.0000000f,   0.1250000f},
	{"small_pos_full",               0.0000000f,   0.8437000f},
	{"small_neg_full",              -0.8437000f,   0.0000000f},
	{"small_mixed_full",            -0.8437000f,   0.8437000f},

	{"small_centered_tight",        -0.1250000f,   0.1250000f},
	{"small_centered_medium",       -0.5000000f,   0.5000000f},

	{"boundary_0p84375_tight_pos",   0.8430000f,   0.8445000f},
	{"boundary_0p84375_tight_neg",  -0.8445000f,  -0.8430000f},
	{"boundary_0p84375_mixed",      -0.8445000f,   0.8445000f},
	{"boundary_0p84375_wide_mixed", -0.8600000f,   0.8600000f},

	{"mid_pos_full",                 0.8437500f,   1.2499000f},
	{"mid_neg_full",                -1.2499000f,  -0.8437500f},
	{"mid_mixed_full",              -1.2499000f,   1.2499000f},

	{"mid_pos_inner",                0.9000000f,   1.2000000f},
	{"mid_neg_inner",               -1.2000000f,  -0.9000000f},

	{"boundary_1p25_tight_pos",      1.2450000f,   1.2550000f},
	{"boundary_1p25_tight_neg",     -1.2550000f,  -1.2450000f},
	{"boundary_1p25_mixed",         -1.2550000f,   1.2550000f},
	{"boundary_1p25_wide_mixed",    -1.3000000f,   1.3000000f},

	{"tail1_pos_full",               1.2500000f,   2.8570000f},
	{"tail1_neg_full",              -2.8570000f,  -1.2500000f},
	{"tail1_mixed_full",            -2.8570000f,   2.8570000f},

	{"tail1_pos_inner",              1.4000000f,   2.5000000f},
	{"tail1_neg_inner",             -2.5000000f,  -1.4000000f},

	{"tail1_pos_common",             1.0000000f,   3.0000000f},

	{"boundary_2p857143_tight_pos",  2.8500000f,   2.8650000f},
	{"boundary_2p857143_tight_neg", -2.8650000f,  -2.8500000f},
	{"boundary_2p857143_mixed",     -2.8650000f,   2.8650000f},
	{"boundary_2p857143_wide_mixed",-2.9000000f,   2.9000000f},

	{"tail2_pos_full",               2.8571430f,   9.9990000f},
	{"tail2_neg_full",              -9.9990000f,  -2.8571430f},
	{"tail2_mixed_full",            -9.9990000f,   9.9990000f},

	{"tail2_pos_inner",              3.0000000f,   8.0000000f},
	{"tail2_neg_inner",             -8.0000000f,  -3.0000000f},

	{"tail2_pos_common",             3.0000000f,   6.0000000f},
	{"tail2_pos_far",                6.0000000f,   9.9990000f},

	{"boundary_10_tight_pos",        9.9500000f,  10.0500000f},
	{"boundary_10_tight_neg",      -10.0500000f,  -9.9500000f},
	{"boundary_10_mixed",          -10.0500000f,  10.0500000f},
	{"boundary_10_wide_mixed",     -10.5000000f,  10.5000000f},

	{"huge_pos_small_over_10",      10.0000000f,  12.0000000f},
	{"huge_pos_far",                12.0000000f,  50.0000000f},
	{"huge_neg_small_over_10",     -12.0000000f, -10.0000000f},
	{"huge_neg_far",               -50.0000000f, -12.0000000f},
	{"huge_mixed_full",            -50.0000000f,  50.0000000f},

	{"mixed_small_mid",             -1.0000000f,   1.0000000f},
	{"mixed_mid_tail1",             -2.0000000f,   2.0000000f},
	{"mixed_small_mid_tail1",       -3.0000000f,   3.0000000f},
	{"mixed_tail1_tail2",           -4.0000000f,   4.0000000f},
	{"mixed_all_finite_branches",  -10.0000000f,  10.0000000f},
	{"mixed_all_with_huge",        -12.0000000f,  12.0000000f},

	{"realistic_centered_narrow",   -1.5000000f,   1.5000000f},
	{"realistic_centered_unit",     -2.0000000f,   2.0000000f},
	{"realistic_centered_wide",     -4.0000000f,   4.0000000f},
	{"realistic_positive_core",      0.0000000f,   3.0000000f},
	{"realistic_positive_wide",      0.0000000f,   6.0000000f},
	{"realistic_positive_far_tail",  2.0000000f,  10.0000000f}
};

std::vector<std::tuple<std::string, float, float>> sinh_fp32_test_range = {
	{"tiny_pos_tight_near_zero",      0.0000000f,   0.0312500f},
	{"tiny_neg_tight_near_zero",     -0.0312500f,   0.0000000f},
	{"tiny_mixed_near_zero",         -0.0312500f,   0.0312500f},
	{"activation_like_tiny",         -0.1250000f,   0.1250000f},
	{"activation_like_small",        -0.5000000f,   0.5000000f},
	{"small_pos_full_poly",           0.0000000f,   0.9990000f},
	{"small_neg_full_poly",          -0.9990000f,   0.0000000f},
	{"small_mixed_full_poly",        -0.9990000f,   0.9990000f},
	{"boundary_1p0_tight_pos",        0.9950000f,   1.0050000f},
	{"boundary_1p0_tight_neg",       -1.0050000f,  -0.9950000f},
	{"boundary_1p0_tight_mixed",     -1.0050000f,   1.0050000f},
	{"normalized_centered",          -1.0000000f,   1.0000000f},
	{"common_centered_small",        -2.0000000f,   2.0000000f},
	{"mid_mixed_exp_path",           -3.0000000f,   3.0000000f},
	{"common_centered_wide",         -5.0000000f,   5.0000000f},
	{"stress_centered_10",          -10.0000000f,  10.0000000f},
	{"positive_only_small",           0.0000000f,   3.0000000f},
	{"positive_only_wide",            0.0000000f,  10.0000000f},
	{"negative_only_wide",          -10.0000000f,   0.0000000f},
	{"large_finite_positive",        10.0000000f,  40.0000000f},
	{"large_finite_negative",       -40.0000000f, -10.0000000f},
	{"large_finite_centered",       -40.0000000f,  40.0000000f},
	{"very_large_mixed_finite",     -80.0000000f,  80.0000000f},
	{"near_overflow_positive",       80.0000000f,  89.3000000f},
	{"near_overflow_negative",      -89.3000000f, -80.0000000f},
	{"near_overflow_centered",      -89.3000000f,  89.3000000f},
	{"boundary_overflow_tight_pos",  89.3000000f,  89.5000000f},
	{"boundary_overflow_tight_neg", -89.5000000f, -89.3000000f},
	{"boundary_overflow_tight_mixed",-89.5000000f,  89.5000000f},
	{"overflow_pos_only",            89.5000000f, 100.0000000f},
	{"overflow_neg_only",          -100.0000000f, -89.5000000f},
	{"overflow_mixed",             -100.0000000f, 100.0000000f}
};


std::vector<std::tuple<std::string, float, float>> cosh_fp32_test_range = {
	{"tiny_pos_tight_near_zero",      0.0000000f,   0.0312500f},
	{"tiny_mixed_near_zero",         -0.0312500f,   0.0312500f},
	{"activation_like_tiny",         -0.1250000f,   0.1250000f},
	{"activation_like_small",        -0.5000000f,   0.5000000f},
	{"small_pos_full_poly",           0.0000000f,   0.9990000f},
	{"small_mixed_full_poly",        -0.9990000f,   0.9990000f},
	{"boundary_1p0_tight_pos",        0.9950000f,   1.0050000f},
	{"boundary_1p0_tight_mixed",     -1.0050000f,   1.0050000f},
	{"normalized_centered",          -1.0000000f,   1.0000000f},
	{"common_centered_small",        -2.0000000f,   2.0000000f},
	{"mid_mixed_exp_path",           -3.0000000f,   3.0000000f},
	{"common_centered_wide",         -5.0000000f,   5.0000000f},
	{"stress_centered_10",          -10.0000000f,  10.0000000f},
	{"nonnegative_small",             0.0000000f,   3.0000000f},
	{"nonnegative_wide",              0.0000000f,  10.0000000f},
	{"large_finite_nonnegative",     10.0000000f,  40.0000000f},
	{"large_finite_centered",       -40.0000000f,  40.0000000f},
	{"very_large_mixed_finite",     -80.0000000f,  80.0000000f},
	{"near_overflow_nonnegative",    80.0000000f,  89.3000000f},
	{"near_overflow_centered",      -89.3000000f,  89.3000000f},
	{"boundary_overflow_tight_pos",  89.3000000f,  89.5000000f},
	{"boundary_overflow_tight_mixed",-89.5000000f,  89.5000000f},
	{"overflow_pos_only",            89.5000000f, 100.0000000f},
	{"overflow_mixed",             -100.0000000f, 100.0000000f}
};


std::vector<std::tuple<std::string, float, float>> tanh_fp32_test_range = {
	{"tiny_pos_tight_near_zero",       0.0000000f,   0.0312500f},
	{"tiny_neg_tight_near_zero",      -0.0312500f,   0.0000000f},
	{"tiny_mixed_near_zero",          -0.0312500f,   0.0312500f},
	{"activation_like_tiny",          -0.1250000f,   0.1250000f},
	{"activation_like_small",         -0.5000000f,   0.5000000f},
	{"small_pos_full_poly",            0.0000000f,   0.6240000f},
	{"small_neg_full_poly",           -0.6240000f,   0.0000000f},
	{"small_mixed_full_poly",         -0.6240000f,   0.6240000f},
	{"boundary_0p625_tight_pos",       0.6200000f,   0.6300000f},
	{"boundary_0p625_tight_neg",      -0.6300000f,  -0.6200000f},
	{"boundary_0p625_tight_mixed",    -0.6300000f,   0.6300000f},
	{"normalized_centered",           -1.0000000f,   1.0000000f},
	{"common_centered_small",         -2.0000000f,   2.0000000f},
	{"common_centered_medium",        -3.0000000f,   3.0000000f},
	{"common_centered_wide",          -5.0000000f,   5.0000000f},
	{"positive_only_small",            0.0000000f,   3.0000000f},
	{"positive_only_wide",             0.0000000f,  10.0000000f},
	{"negative_only_wide",           -10.0000000f,   0.0000000f},
	{"boundary_10_tight_pos",          9.9000000f,  10.1000000f},
	{"boundary_10_tight_neg",        -10.1000000f,  -9.9000000f},
	{"boundary_10_tight_mixed",      -10.1000000f,  10.1000000f},
	{"saturation_transition_pos",      5.0000000f,  10.0000000f},
	{"saturation_transition_neg",    -10.0000000f,  -5.0000000f},
	{"saturation_transition_mixed",  -10.0000000f,  10.0000000f},
	{"large_saturated_positive",      10.0000000f,  20.0000000f},
	{"large_saturated_negative",     -20.0000000f, -10.0000000f},
	{"large_saturated_centered",     -20.0000000f,  20.0000000f},
	{"very_large_saturated_positive", 20.0000000f,  80.0000000f},
	{"very_large_saturated_negative",-80.0000000f, -20.0000000f},
	{"very_large_saturated_centered",-80.0000000f,  80.0000000f}
};


std::vector<std::tuple<std::string, float, float>> atan_fp32_test_range = {
	{"tiny_pos_full",                 0.0000000f,    0.0002440f},
	{"tiny_mixed_full",              -0.0002440f,    0.0002440f},

	{"boundary_2powm12_tight_pos",    0.0002400f,    0.0002485f},
	{"boundary_2powm12_mixed",       -0.0002485f,    0.0002485f},

	{"small_pos_full",                0.0002442f,    0.4374000f},
	{"small_mixed_full",             -0.4374000f,    0.4374000f},

	{"boundary_0p4375_tight_pos",     0.4360000f,    0.4390000f},
	{"boundary_0p4375_mixed",        -0.4390000f,    0.4390000f},

	{"region0_pos_full",              0.4375000f,    0.6874000f},
	{"region0_mixed_full",           -0.6874000f,    0.6874000f},

	{"boundary_0p6875_tight_pos",     0.6850000f,    0.6900000f},
	{"boundary_0p6875_mixed",        -0.6900000f,    0.6900000f},

	{"region1_pos_full",              0.6875000f,    1.1874000f},
	{"region1_mixed_full",           -1.1874000f,    1.1874000f},

	{"boundary_1p1875_tight_pos",     1.1800000f,    1.1950000f},
	{"boundary_1p1875_mixed",        -1.1950000f,    1.1950000f},

	{"region2_pos_full",              1.1875000f,    2.4374000f},
	{"region2_mixed_full",           -2.4374000f,    2.4374000f},

	{"boundary_2p4375_tight_pos",     2.4200000f,    2.4500000f},
	{"boundary_2p4375_mixed",        -2.4500000f,    2.4500000f},

	{"region3_pos_full",              2.4375000f,   10.0000000f},
	{"region3_mixed_full",          -10.0000000f,   10.0000000f},
	{"region3_pos_far",               8.0000000f,   50.0000000f},

	{"mixed_tiny_small",             -0.0010000f,    0.0010000f},
	{"mixed_small_region0",          -0.7000000f,    0.7000000f},
	{"mixed_region0_region1",        -1.0000000f,    1.0000000f},
	{"mixed_region1_region2",        -2.0000000f,    2.0000000f},
	{"mixed_region2_region3",        -3.0000000f,    3.0000000f},
	{"mixed_all_finite_branches",   -10.0000000f,   10.0000000f},

	{"realistic_centered_tiny",      -0.1000000f,    0.1000000f},
	{"realistic_centered_unit",      -1.0000000f,    1.0000000f},
	{"realistic_centered_wide",      -4.0000000f,    4.0000000f},
	{"realistic_positive_unit",       0.0000000f,    1.0000000f},
	{"realistic_positive_wide",       0.0000000f,    4.0000000f}
};

std::vector<std::tuple<std::string, float, float>> asinh_fp32_test_range = {
	{"tiny_pos_full",                 0.0000000000f,   0.0000000037f},
	{"tiny_mixed_full",              -0.0000000037f,   0.0000000037f},

	{"boundary_2powm28_tight_pos",    0.0000000035f,   0.0000000039f},
	{"boundary_2powm28_mixed",       -0.0000000039f,   0.0000000039f},

	{"small_pos_near_zero",           0.0000000038f,   0.0312500000f},
	{"small_pos_full",                0.0000000038f,   1.9999000000f},
	{"small_neg_full",               -1.9999000000f,  -0.0000000038f},
	{"small_mixed_full",             -1.9999000000f,   1.9999000000f},

	{"small_centered_tight",         -0.1250000000f,   0.1250000000f},
	{"small_centered_medium",        -0.5000000000f,   0.5000000000f},
	{"small_centered_wide",          -1.0000000000f,   1.0000000000f},
	{"small_positive_unit",           0.0000000000f,   1.0000000000f},

	{"boundary_2_tight_pos",          1.9500000000f,   2.0500000000f},
	{"boundary_2_tight_neg",         -2.0500000000f,  -1.9500000000f},
	{"boundary_2_mixed",             -2.0500000000f,   2.0500000000f},
	{"boundary_2_wide_mixed",        -2.2500000000f,   2.2500000000f},

	{"medium_pos_full",               2.0001000000f,   1000.0000000f},
	{"medium_neg_full",             -1000.0000000f,  -2.0001000000f},
	{"medium_mixed_full",           -1000.0000000f,   1000.0000000f},

	{"medium_pos_inner",              3.0000000000f,   100.0000000f},
	{"medium_neg_inner",            -100.0000000f,   -3.0000000000f},

	{"medium_pos_far",             1000.0000000f,   10000000.0000000f},
	{"medium_mixed_far",         -10000000.0000000f, 10000000.0000000f},

	{"boundary_2pow28_tight_pos",   260000000.0000000f, 276000000.0000000f},
	{"boundary_2pow28_tight_neg",  -276000000.0000000f,-260000000.0000000f},
	{"boundary_2pow28_mixed",      -276000000.0000000f, 276000000.0000000f},

	{"large_pos_small_over",        268435456.0000000f, 1000000000.0000000f},
	{"large_neg_small_over",      -1000000000.0000000f,-268435456.0000000f},
	{"large_mixed_full",          -1000000000.0000000f, 1000000000.0000000f},
	{"large_pos_far",              1000000000.0000000f, 3.4028235e38f},

	{"mixed_tiny_small",             -0.0000010000f,   0.0000010000f},
	{"mixed_small_medium",           -4.0000000000f,   4.0000000000f},
	{"mixed_small_to_large",      -1000000000.0000000f, 1000000000.0000000f},

	{"realistic_centered_unit",      -1.0000000000f,   1.0000000000f},
	{"realistic_centered_wide",      -4.0000000000f,   4.0000000000f},
	{"realistic_positive_wide",       0.0000000000f,   8.0000000000f}
};

std::vector<std::tuple<std::string, float, float>> tan_fp32_test_range = {
	{"tiny_pos_tight_near_zero",          0.0000000f,     0.0312500f},
	{"tiny_mixed_near_zero",             -0.0312500f,     0.0312500f},

	{"activation_like_small",            -0.1250000f,     0.1250000f},
	{"poly_fast_full_pio8_pos",           0.0000000f,     0.3920000f},
	{"poly_fast_full_pio8_mixed",        -0.3920000f,     0.3920000f},

	{"boundary_pio8_tight_pos",           0.3900000f,     0.3955000f},
	{"boundary_pio8_tight_neg",          -0.3955000f,    -0.3900000f},
	{"boundary_pio8_tight_mixed",        -0.3955000f,     0.3955000f},

	{"transform_region_pio8_to_pio4_pos", 0.3926991f,     0.7850000f},
	{"transform_region_pio8_to_pio4_mixed",-0.7850000f,    0.7850000f},

	{"boundary_pio4_tight_pos",           0.7800000f,     0.7900000f},
	{"boundary_pio4_tight_mixed",        -0.7900000f,     0.7900000f},

	{"near_pio2_left_tight",              1.5200000f,     1.5695000f},
	{"near_pio2_right_tight",             1.5719000f,     1.6200000f},
	{"near_pio2_crossing",                1.5200000f,     1.6200000f},

	{"centered_one_quadrant",            -1.0000000f,     1.0000000f},
	{"centered_two_quadrants",           -2.0000000f,     2.0000000f},
	{"centered_four_quadrants",          -3.2000000f,     3.2000000f},

	{"boundary_pi_tight_mixed",           3.1000000f,     3.1800000f},
	{"around_pi_centered",               -3.2000000f,     3.2000000f},

	{"near_3pio2_left_tight",             4.6600000f,     4.7115000f},
	{"near_3pio2_right_tight",            4.7132000f,     4.7650000f},
	{"near_3pio2_crossing",               4.6600000f,     4.7650000f},

	{"moderate_positive_wide",            0.0000000f,    10.0000000f},
	{"moderate_negative_wide",          -10.0000000f,     0.0000000f},
	{"moderate_centered_wide",          -10.0000000f,    10.0000000f},

	{"huge_fast_edge_below_2pow23",  8380000.0000000f, 8388607.0000000f},
	{"threshold_2pow23_crossing",    8388500.0000000f, 8388700.0000000f},
	{"slow_path_all_above_2pow23",   8388609.0000000f, 9000000.0000000f}
};

std::vector<std::tuple<std::string, float, float>> acosh_fp32_test_range = {
	{"invalid_all_negative_large",        -1000.0000000f,   -1.0000000f},
	{"invalid_all_negative_small",           -1.0000000f,   -0.0000010f},
	{"invalid_zero_to_one",                   0.0000000f,    0.9999990f},
	{"domain_edge_cross_1_tight",             0.9995000f,    1.0005000f},
	{"domain_edge_cross_1_wide",              0.9000000f,    1.1000000f},
	{"just_above_1_ultra_tight",              1.0000000f,    1.0000100f},
	{"just_above_1_very_tight",               1.0000000f,    1.0001000f},
	{"just_above_1_tight",                    1.0000000f,    1.0010000f},
	{"just_above_1_small",                    1.0000000f,    1.0100000f},
	{"just_above_1_medium",                   1.0000000f,    1.0500000f},
	{"log1p_smallrange_pure",                 1.0000000f,    1.0700000f},
	{"log1p_smallrange_boundary_tight",       1.0680000f,    1.0740000f},
	{"log1p_general_subrange",                1.0715000f,    1.2500000f},
	{"one_to_two_full",                       1.0000000f,    2.0000000f},
	{"one_to_two_upper_half",                 1.5000000f,    2.0000000f},
	{"one_to_two_centered",                   1.2000000f,    1.8000000f},
	{"boundary_2_left_tight",                 1.9500000f,    1.9999000f},
	{"boundary_2_right_tight",                2.0001000f,    2.0500000f},
	{"boundary_2_cross_tight",                1.9900000f,    2.0100000f},
	{"boundary_2_cross_wide",                 1.7500000f,    2.2500000f},
	{"moderate_gt2_small",                    2.0000000f,    4.0000000f},
	{"moderate_gt2_common",                   2.0000000f,   10.0000000f},
	{"moderate_gt2_wide",                     2.0000000f,  100.0000000f},
	{"large_positive_mainpath",              10.0000000f,1000000.0000000f},
	{"very_large_mainpath",             1000000.0000000f,100000000.0000000f},
	{"common_ml_like_near_one",               1.0000000f,    3.0000000f},
	{"common_geometry_like",                  1.0000000f,   10.0000000f},
	{"common_scientific_moderate",            1.0000000f,  100.0000000f},
	{"common_scientific_wide",                1.0000000f,1000000.0000000f},
	{"threshold_2pow28_left_tight",   268430000.0000000f,268435455.0000000f},
	{"threshold_2pow28_right_tight",  268435456.0000000f,268440000.0000000f},
	{"threshold_2pow28_cross_tight",  268435300.0000000f,268435700.0000000f},
	{"threshold_2pow28_cross_wide",   268000000.0000000f,269000000.0000000f},
	{"huge_fastpath_small_span",      268435456.0000000f,300000000.0000000f},
	{"huge_fastpath_mid",             300000000.0000000f,10000000000.0000000f},
	{"huge_fastpath_wide",           1000000000.0000000f,100000002004087734272.0000000f},
	{"mixed_invalid_to_small_valid",         0.5000000f,    1.5000000f},
	{"mixed_invalid_to_gt2",                 0.5000000f,    4.0000000f},
	{"mixed_one_to_gt2_cross_2",             1.0000000f,    4.0000000f},
	{"mixed_small_valid_to_mainpath",        1.0000000f,   10.0000000f},
	{"mixed_mainpath_to_hugepath",           2.0000000f,300000000.0000000f},
	{"mixed_all_valid_paths",                1.0000000f,300000000.0000000f},
	{"mixed_invalid_valid_huge",            -1.0000000f,300000000.0000000f},
	{"centered_small_mixed",                -2.0000000f,    2.0000000f},
	{"centered_moderate_mixed",            -10.0000000f,   10.0000000f},
	{"centered_large_mixed",             -1000.0000000f, 1000.0000000f},
	{"positive_wide_small_to_mid",            1.0000000f,  100.0000000f},
	{"positive_wide_mid_to_large",          100.0000000f,1000000.0000000f},
	{"positive_wide_full_valid_no_huge",      1.0000000f,100000000.0000000f},
	{"positive_wide_full_valid_with_huge",    1.0000000f,1000000000.0000000f}
};

std::vector<std::tuple<std::string, float, float>> sin_fp32_test_range = {
	{"tiny_pos_tight_near_zero",          0.0000000f,      0.0312500f},
	{"tiny_neg_tight_near_zero",         -0.0312500f,      0.0000000f},
	{"tiny_mixed_near_zero",             -0.0312500f,      0.0312500f},
	{"small_activation_like",            -0.1250000f,      0.1250000f},
	{"small_centered",                   -0.5000000f,      0.5000000f},
	{"common_centered_1",                -1.0000000f,      1.0000000f},
	{"common_centered_pi_over_2",        -1.5707964f,      1.5707964f},
	{"boundary_pi_over_2_tight",          1.5207964f,      1.6207964f},
	{"boundary_neg_pi_over_2_tight",     -1.6207964f,     -1.5207964f},
	{"common_centered_pi",               -3.1415927f,      3.1415927f},
	{"boundary_pi_tight",                 3.0915928f,      3.1915927f},
	{"boundary_neg_pi_tight",            -3.1915927f,     -3.0915928f},
	{"common_centered_2pi",              -6.2831855f,      6.2831855f},
	{"boundary_2pi_tight",                6.2331853f,      6.3331857f},
	{"boundary_neg_2pi_tight",           -6.3331857f,     -6.2331853f},
	{"common_centered_10pi",            -31.4159260f,     31.4159260f},
	{"positive_only_10pi",                0.0000000f,     31.4159260f},
	{"negative_only_10pi",              -31.4159260f,      0.0000000f},
	{"wide_common_100pi",              -314.1592712f,    314.1592712f},
	{"positive_only_100pi",               0.0000000f,    314.1592712f},
	{"negative_only_100pi",            -314.1592712f,      0.0000000f},
	{"large_fastpath_centered_1e4",   -10000.0000000f,  10000.0000000f},
	{"large_fastpath_centered_1e6", -1000000.0000000f,1000000.0000000f},
	{"fastpath_boundary_below_tight", 8380000.0000000f, 8388607.0000000f},
	{"slowpath_boundary_cross_mixed",  8388000.0000000f, 8389200.0000000f},
	{"slowpath_boundary_cross_neg",   -8389200.0000000f,-8388000.0000000f},
	{"slowpath_positive_only",         8388609.0000000f,12000000.0000000f},
	{"slowpath_negative_only",       -12000000.0000000f,-8388609.0000000f},
	{"slowpath_centered_mixed",      -12000000.0000000f,12000000.0000000f}
};

std::vector<std::tuple<std::string, float, float>> cos_fp32_test_range = {
	{"tiny_pos_tight_near_zero",          0.0000000f,      0.0312500f},
	{"tiny_neg_tight_near_zero",         -0.0312500f,      0.0000000f},
	{"tiny_mixed_near_zero",             -0.0312500f,      0.0312500f},
	{"small_activation_like",            -0.1250000f,      0.1250000f},
	{"small_centered",                   -0.5000000f,      0.5000000f},
	{"common_centered_1",                -1.0000000f,      1.0000000f},
	{"common_centered_pi_over_2",        -1.5707964f,      1.5707964f},
	{"boundary_pi_over_2_tight",          1.5207964f,      1.6207964f},
	{"boundary_neg_pi_over_2_tight",     -1.6207964f,     -1.5207964f},
	{"common_centered_pi",               -3.1415927f,      3.1415927f},
	{"boundary_pi_tight",                 3.0915928f,      3.1915927f},
	{"boundary_neg_pi_tight",            -3.1915927f,     -3.0915928f},
	{"common_centered_2pi",              -6.2831855f,      6.2831855f},
	{"boundary_2pi_tight",                6.2331853f,      6.3331857f},
	{"boundary_neg_2pi_tight",           -6.3331857f,     -6.2331853f},
	{"common_centered_10pi",            -31.4159260f,     31.4159260f},
	{"common_centered_100pi",          -314.1592712f,    314.1592712f},
	{"positive_only_100pi",               0.0000000f,    314.1592712f},
	{"negative_only_100pi",            -314.1592712f,      0.0000000f},
	{"large_fastpath_centered_1e4",   -10000.0000000f,  10000.0000000f},
	{"large_fastpath_centered_1e6", -1000000.0000000f,1000000.0000000f},
	{"fastpath_boundary_below_tight", 8380000.0000000f, 8388607.0000000f},
	{"fastpath_boundary_below_mixed", -8388607.0000000f, 8388607.0000000f},
	{"slowpath_boundary_cross_pos",    8388000.0000000f, 8389200.0000000f},
	{"slowpath_boundary_cross_neg",   -8389200.0000000f,-8388000.0000000f},
	{"slowpath_boundary_cross_mixed", -8389200.0000000f, 8389200.0000000f},
	{"slowpath_positive_only",         8388609.0000000f,12000000.0000000f},
	{"slowpath_negative_only",       -12000000.0000000f,-8388609.0000000f},
	{"slowpath_centered_mixed",      -12000000.0000000f,12000000.0000000f}
};

std::vector<std::tuple<std::string, float, float>> log_fp32_test_range = {
	{"subnormal_tiny_pos_full",              0.0000000f,        1.1754942e-38f},
	{"subnormal_tiny_pos_tight",             1.4012985e-45f,    1.0000000e-39f},
	{"subnormal_to_min_normal_cross",        1.0000000e-45f,    1.3000000e-38f},
	{"min_normal_boundary_tight",            1.1000000e-38f,    1.2500000e-38f},

	{"very_small_pos_log_heavy",             1.0000000e-30f,    1.0000000e-20f},
	{"small_pos_prob_like",                  1.0000000e-12f,    1.0000000e-6f},
	{"small_pos_common",                     1.0000000e-6f,     1.0000000e-2f},
	{"fractional_pos_below_one",             1.0000000e-2f,     5.0000000e-1f},
	{"fractional_pos_wide",                  1.0000000e-3f,     9.9999994e-1f},

	{"near1_ultra_tight",                    0.9990000f,        1.0010000f},
	{"near1_very_tight",                     0.9900000f,        1.0100000f},
	{"near1_tight",                          0.9500000f,        1.0500000f},
	{"near1_full_branch_window",             0.7501000f,        1.2499000f},
	{"near1_lower_boundary_tight",           0.7400000f,        0.7600000f},
	{"near1_upper_boundary_tight",           1.2400000f,        1.2600000f},

	{"around_sqrt_half_boundary_tight",      0.6970000f,        0.7170000f},
	{"pos_unit_to_small_scale",              0.5000000f,        2.0000000f},
	{"common_positive_centered_1_10",        1.0000000f,       10.0000000f},
	{"common_positive_centered_1e2",         1.0000000f,      100.0000000f},
	{"common_positive_centered_1e4",         1.0000000f,    10000.0000000f},
	{"mixed_scale_positive_1e_6_to_1e6",     0.0000010f,   1000000.0000000f},

	{"large_positive_1e6_to_1e12",           1000000.0000000f,      1.0000000e12f},
	{"huge_positive_1e12_to_1e20",           1.0000000e12f,          1.0000000e20f},
	{"huge_positive_1e20_to_1e30",           1.0000000e20f,          1.0000000e30f},
	{"max_finite_boundary_tight",            1.0000000e37f,          3.4028233e38f},

	{"zero_and_tiny_pos_mixed",              0.0000000f,        1.0000000e-6f},
	{"negative_tiny_only",                  -1.0000000e-6f,    0.0000000f},
	{"negative_fractional_only",            -1.0000000f,       -0.0000010f},
	{"negative_large_only",                 -1000000.0000000f, -1.0000000f},
	{"mixed_invalid_and_valid_small",       -1.0000000f,        1.0000000f},

	{"probability_like_open",                 1.0000000e-8f,     1.0000000f},
	{"softmax_tail_like",                     1.0000000e-12f,    1.0000000e-2f},
	{"loss_input_like",                       1.0000000e-4f,     1.0000000e1f},
};

std::vector<std::tuple<std::string, float, float>> atanh_fp32_test_range = {
	{"tiny_signed_linear_ultra",              -3.7252903e-09f,   3.7252903e-09f},
	{"tiny_signed_linear_tight",              -1.0000000e-08f,   1.0000000e-08f},
	{"tiny_signed_linear_wide",               -1.0000000e-06f,   1.0000000e-06f},

	{"small_signed_below_half_1e4",           -1.0000000e-04f,   1.0000000e-04f},
	{"small_signed_below_half_1e2",           -1.0000000e-02f,   1.0000000e-02f},
	{"small_signed_below_half_common",        -1.0000000e-01f,   1.0000000e-01f},
	{"mid_signed_below_half_full",            -4.9990000e-01f,   4.9990000e-01f},

	{"half_boundary_ultra_tight",              4.9990000e-01f,   5.0010000e-01f},
	{"half_boundary_tight_signed",            -5.0100000e-01f,   5.0100000e-01f},
	{"half_to_three_quarters_pos",             5.0000000e-01f,   7.5000000e-01f},
	{"half_to_three_quarters_signed",         -7.5000000e-01f,   7.5000000e-01f},

	{"moderate_signed_common",                -7.5000000e-01f,   7.5000000e-01f},
	{"moderate_pos_activation_like",           1.0000000e-02f,   8.0000000e-01f},
	{"moderate_signed_activation_like",       -8.0000000e-01f,   8.0000000e-01f},

	{"near_one_loose_pos",                     8.0000000e-01f,   9.5000000e-01f},
	{"near_one_loose_signed",                 -9.5000000e-01f,   9.5000000e-01f},
	{"near_one_tight_pos",                     9.5000000e-01f,   9.9000000e-01f},
	{"near_one_tight_signed",                 -9.9000000e-01f,   9.9000000e-01f},
	{"near_one_very_tight_pos",                9.9000000e-01f,   9.9900000e-01f},
	{"near_one_very_tight_signed",            -9.9900000e-01f,   9.9900000e-01f},
	{"near_one_ultra_tight_pos",               9.9900000e-01f,   9.9999000e-01f},
	{"near_one_ultra_tight_signed",           -9.9999000e-01f,   9.9999000e-01f},

	{"domain_edge_inside_pos",                 9.9990000e-01f,   9.9999994e-01f},
	{"domain_edge_inside_signed",             -9.9999994e-01f,   9.9999994e-01f}
};

std::vector<std::tuple<std::string, float, float>> asin_fp32_test_range =
{
	{"domain",                       -1.0f,     1.0f}
};

//int main()
//{
//	mkl_set_num_threads(1);
//
//	{
//		constexpr std::size_t count = 1 << 23;
//
//		run_1in_1out_test<float, count, 32>(
//			asinh_fp32_test_range,
//			[](std::size_t n, const float* in, float* out) -> void { vmsAsinh(n, in, out, VML_EP); },
//			[](std::size_t n, const float* in, float* out) -> void { fy::asinh(n, in, out); }
//		);
//	}
//	system("pause");
//	{
//		constexpr std::size_t count = 1 << 23;
//
//		run_1in_1out_test<float, count, 32>(
//			asinh_fp32_test_range,
//			[](std::size_t n, const float* in, float* out) -> void { vmsAcosh(n, in, out, VML_EP); },
//			[](std::size_t n, const float* in, float* out) -> void { fy::acosh(n, in, out); }
//		);
//	}
//}




template<typename before_expr, typename after_expr,
	typename reference_expr, typename test_expr>
void run_1in_1out_domain_fp32_test(
	before_expr&& before, after_expr&& after,
	reference_expr&& reference_func, test_expr&& test_func);


int main()
{
	mkl_set_num_threads(1);

	auto ulp_distance = [](float a, float b) -> std::size_t
		{
			return interval_test_1in_1out_invoker<float>::acquired_ulp(a, b);
		};



	constexpr std::size_t batch_size = 1 << 23;
	float* workspace_re = reinterpret_cast<float*>(_aligned_malloc(sizeof(float) * batch_size, 32));
	float* workspace_te = reinterpret_cast<float*>(_aligned_malloc(sizeof(float) * batch_size, 32));

	interval_bit_pattern_iteration_invoker invoker(batch_size, -1.0f, 1.0f);

	std::size_t max_ulp{ 0 };
	while (invoker.next_batch([&](const float* this_round_begin, std::size_t this_round_length)
		{
			vmsAtanh(batch_size, this_round_begin, workspace_re, VML_HA);
			fy::atanh(batch_size, this_round_begin, workspace_te);

			std::size_t this_round_ulp{ 0 };
			for (std::size_t i = 0; i < this_round_length; ++i)
			{
				this_round_ulp = std::max(this_round_ulp,
					ulp_distance(workspace_re[i], workspace_te[i]));
			}
			if (this_round_ulp >= 4.0)
			{
				system("pause");
			}

			std::cout << std::format("begin: 0x{:08X}, end: 0x{:08X}, max ulp: {}, length: {}, remaining: {}",
				std::bit_cast<uint32_t>(*this_round_begin),
				std::bit_cast<uint32_t>(this_round_begin[this_round_length - 1]),
				this_round_ulp,
				this_round_length,
				invoker.remaining_count()
			) << std::endl;


			max_ulp = std::max(max_ulp, this_round_ulp);
		}))
	{
	}

	/*run_1in_1out_domain_fp32_test(
		[]() -> void { timeBeginPeriod(1); },
		[]() -> void { timeEndPeriod(1); },
		[](std::size_t n, const double* in, double* out) -> void { vmdAsinh(n, in, out, VML_HA);},
		[](std::size_t n, const float* in, float* out) -> void { fy::asinh(n, in, out);}
	);*/

	system("pause");
	constexpr std::size_t count = 1 << 23;

	run_1in_1out_test<float, count, 32>(
		atanh_fp32_test_range,
		[](std::size_t n, const float* in, float* out) -> void { vmsAtanh(n, in, out, VML_EP); },
		[](std::size_t n, const float* in, float* out) -> void { fy::atanh(n, in, out); }
	);
}

